# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# training

import os
import time
import copy
import threading
import random
import bz2
import pickle
import warnings
import queue
from collections import deque
# import faiss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import psutil

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate, softmax
from .model import to_torch, to_gpu, ModelWrapper
from .losses import compute_target, compute_rnd
from .connection import MultiProcessJobExecutor
from .worker import WorkerCluster, WorkerServer
from .metadata import KNN, feed_knn


def make_batch(episodes, step_init, step_length, args, 
               keys={'moment': 'moment', 'start':'start', 'end':'end', 'base':'base', 'train_start':'train_start'}
               ):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []
    k = keys

    def replace_none(a, b):
        return a if a is not None else b

    for i in range(len(episodes[0])):
        for ep_ in episodes:
            ep = ep_[i]
            moments_ = sum([pickle.loads(bz2.decompress(ms)) for ms in ep[k['moment']]], [])
            moments = moments_[ep[k['start']] - ep[k['base']]:ep[k['end']] - ep[k['base']]]
            players = list(moments[0]['observation'].keys())
            if not args['turn_based_training']:  # solo training
                players = [random.choice(players)]

            # template for padding
            obs_zeros = map_r(moments[0]['observation'][moments[0]['turn'][0]], lambda o: np.zeros_like(o))
            amask_zeros = np.zeros_like(moments[0]['action_mask'][moments[0]['turn'][0]])

            # data that is changed by training configuration
            if args['turn_based_training'] and not args['observation']:
                obs = [[m['observation'][m['turn'][0]]] for m in moments]
                prob = np.array([[[m['selected_prob'][m['turn'][0]]]] for m in moments])
                act = np.array([[m['action'][m['turn'][0]]] for m in moments], dtype=np.int64)[..., np.newaxis]
                amask = np.array([[m['action_mask'][m['turn'][0]]] for m in moments])
            else:
                obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
                prob = np.array([[[replace_none(m['selected_prob'][player], 1.0)] for player in players] for m in moments])
                act = np.array([[replace_none(m['action'][player], 0) for player in players] for m in moments], dtype=np.int64)[..., np.newaxis]
                amask = np.array([[replace_none(m['action_mask'][player], amask_zeros + 1e32) for player in players] for m in moments])
                c = np.array([[[replace_none(m['c'][player], 0.0)] for player in players] for m in moments])
                c_reg = np.array([[[replace_none(m['c_reg'][player], 0.0)] for player in players] for m in moments])
                entropy_srs = np.array([[[replace_none(m['entropy_srs'][player], 0.0)] for player in players] for m in moments])

            # reshape observation
            obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
            obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))

            # datum that is not changed by training configuration
            v = np.array([[replace_none(m['value'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
            rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
            ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
            oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)
            ter = np.array([[replace_none(m['terminal'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)

            emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
            tmask = np.array([[[m['selected_prob'][player] is not None] for player in players] for m in moments], dtype=np.float32)
            omask = np.array([[[m['observation'][player] is not None] for player in players] for m in moments], dtype=np.float32)

            progress = np.arange(ep[k['start']], ep[k['end']], dtype=np.float32)[..., np.newaxis] / ep['total']

            # pad each array if step length is short
            batch_steps = step_init + step_length
            if len(tmask) < batch_steps:
                pad_len_b = step_init - (ep[k['train_start']] - ep[k['start']])
                pad_len_a = batch_steps - len(tmask) - pad_len_b
                obs = map_r(obs, lambda o: np.pad(o, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
                prob = np.pad(prob, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=1)
                v = np.concatenate([np.pad(v, [(pad_len_b, 0), (0, 0), (0, 0)], 'constant', constant_values=0), np.tile(oc, [pad_len_a, 1, 1])])
                act = np.pad(act, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                rew = np.pad(rew, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                ret = np.pad(ret, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                ter = np.pad(ter, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=1)
                emask = np.pad(emask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                tmask = np.pad(tmask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                omask = np.pad(omask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                amask = np.pad(amask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=1e32)
                progress = np.pad(progress, [(pad_len_b, pad_len_a), (0, 0)], 'constant', constant_values=1)
                c = np.pad(c, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                c_reg = np.pad(c_reg, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
                entropy_srs = np.pad(entropy_srs, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)

            obss.append(obs)
            datum.append((prob, v, act, oc, rew, ret, ter, emask, tmask, omask, amask, progress, c, c_reg, entropy_srs))

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    prob, v, act, oc, rew, ret, ter, emask, tmask, omask, amask, progress, c, c_reg, entropy_srs = [to_torch(np.array(val)) for val in zip(*datum)]

    return {
        'observation': obs,
        'selected_prob': prob,
        'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'terminal': ter,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'action_mask': amask,
        'progress': progress,
        'c': c,
        'c_reg': c_reg,
        'entropy_srs': entropy_srs,
    }


def forward_prediction(model, hidden, batch, args):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of neural network
    """

    observations = batch['observation']  # (..., B, T, P or 1, ...)
    actions = batch['action']  # (..., B, T, P or 1, ...)
    batch_shape = batch['action'].size()[:3]  # (B, T, P or 1)
    if args['agent']['ASC_trajectory_length'] > 0:
        observations_set = batch['set']['observation']  # (..., B, T, P or 1, ...)
        actions_set = batch['set']['action']  # (..., B, T, P or 1, ...)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.flatten(0, 2))  # (..., B * T * P or 1, ...)
        act = map_r(actions, lambda o: o.flatten(0, 2))  # (..., B * T * P or 1, ...)
        inputs = {'o':obs, 'a':act}
        if args['agent']['ASC_trajectory_length'] > 0:
            obs_set = observations_set.squeeze(dim=2) 
            act_set = actions_set.squeeze(dim=2) 
            inputs = {** inputs, **{'os': obs_set, 'as': act_set}}
        outputs = model(inputs)
        outputs = map_r(outputs, lambda o: o.unflatten(0, batch_shape) if o.size(0) != batch_shape[0] else o)  # (..., B, T, P or 1, ...)
    else:
        # sequential computation with RNN
        outputs = {}
        for t in range(batch_shape[1]):
            obs = map_r(observations, lambda o: o[:, t].flatten(0, 1))  # (..., B * P or 1, ...)
            omask_ = batch['observation_mask'][:, t]
            omask = map_r(hidden, lambda h: omask_.view(*h.size()[:2], *([1] * (h.dim() - 2))))
            hidden_ = bimap_r(hidden, omask, lambda h, m: h * m)  # (..., B, P, ...)
            if args['turn_based_training'] and not args['observation']:
                hidden_ = map_r(hidden_, lambda h: h.sum(1))  # (..., B * 1, ...)
            else:
                hidden_ = map_r(hidden_, lambda h: h.flatten(0, 1))  # (..., B * P, ...)
            if t < args['burn_in_steps']:
                model.eval()
                with torch.no_grad():
                    outputs_ = model({'o':obs}, hidden_)
            else:
                if not model.training:
                    model.train()
                outputs_ = model({'o':obs}, hidden_)
            outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2])))  # (..., B, P or 1, ...)
            for k, o in outputs_.items():
                if k == 'hidden':
                    next_hidden = o
                else:
                    outputs[k] = outputs.get(k, []) + [o]
            hidden = trimap_r(hidden, next_hidden, omask, lambda h, nh, m: h * (1 - m) + nh * m)
        outputs = {k: torch.stack(o, dim=1) for k, o in outputs.items() if o[0] is not None}

    for k, o in outputs.items():
        if k == 'policy' or k == 'confidence' or k == 're_policy':
            o = o.mul(batch['turn_mask'])
            if o.size(2) > 1 and batch_shape[2] == 1:  # turn-alternating batch
                o = o.sum(2, keepdim=True)  # gather turn player's policies
            outputs[k] = o - batch['action_mask']
        elif '_set' in k: # 集合に対する処理
            # TODO mask の掛け方を考える
            outputs[k] = o
        else:
            # mask valid target values and cumulative rewards
            outputs[k] = o.mul(batch['observation_mask'])
        # TODO: policy_vq_latent など特殊な形状になりえるベクトルの形状にも対応する 

    return outputs


def lastcut_for_buckup(step_masks, teaminals):
    # teaminals を引いて post terminal state(dummy) について伝搬しなくする
    return step_masks.mul(1.0 - teaminals)


def compose_losses(outputs, log_selected_policies, total_advantages, targets, batch, metadataset, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """
    # return_buckup が True だと　teaminals を引いて post terminal state(dummy) について伝搬しなくする
    # losses の計算で終端 Return を target_values (deque) を最初（最後）に入れているのを mask で無効化
    # tmasks = batch['turn_mask'] if args['return_buckup'] else lastcut_for_buckup(batch['turn_mask'], batch['terminal'])
    tmasks = lastcut_for_buckup(batch['turn_mask'], batch['terminal']) if args['return_buckup'] else batch['turn_mask']
    omasks = batch['observation_mask']
    mixed_c = batch['c']
    c_reg = batch['c_reg']
    entropy_srs = batch['entropy_srs']

    # loss の箱
    losses = {}
    # data 数みたい
    dcnt = tmasks.sum().item()

    # policy loss 
    # total_advantages が既に baseline を引いた advantage になっている
    losses['p'] = (-log_selected_policies * total_advantages).mul(tmasks).sum()
    # value loss 
    if 'value' in outputs:
        # value の二乗和誤差
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks).sum() / 2
    # return loss 
    if 'return' in outputs:
        # return の Huber Loss（outputs に return が含まれない限り計算されない）
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks).sum()
    # q-value loss 
    # if 'advantage_for_q' in outputs:
    #     # advantage_for_q の二乗和誤差
    #     losses['a'] = ((outputs['advantage_for_q'] - targets['advantage_for_q']) ** 2).mul(omasks).sum() / 2
    if 'selected_qvalue' in outputs:
        # value の二乗和誤差
        losses['q'] = ((outputs['selected_qvalue'] - targets['selected_qvalue']) ** 2).mul(omasks).sum() / 2

    # rnd embed state loss
    if 'loss_rnd' in outputs:
        losses['rnd'] = outputs['loss_rnd'].mul(tmasks).sum()

    if 'confidence' in outputs:
    # 信頼度の cross_entropy loss
        b_size = outputs['confidence'].shape[0]
        o_size = outputs['confidence'].shape[1]
        a_size = outputs['confidence'].shape[-1]
        o_c = torch.reshape(outputs['confidence'], (b_size * o_size, a_size))
        t_c = torch.reshape(targets['confidence'], (b_size * o_size, 1)).squeeze_()
        # losses['c'] = F.cross_entropy(outputs['confidence'].squeeze(), targets['confidence'].squeeze(), reduction='none').mul(omasks.squeeze()).sum()
        losses['c'] = torch.reshape(F.cross_entropy(o_c, t_c, reduction='none'), (b_size, o_size, 1, 1)).mul(omasks).sum()
        entropy_c_nn = dist.Categorical(logits=outputs['confidence']).entropy().mul(tmasks.mean(-1))
        # 信頼度割合に関する各種監視変数を追加
        losses['ent_c_nn'] = entropy_c_nn.sum()
        if 'knn' in metadataset:
            entropy_c_reg = -torch.sum(c_reg * torch.log(c_reg),4)
            entropy_c_reg = torch.nan_to_num(entropy_c_reg)
            entropy_c_reg = entropy_c_reg.mul(tmasks)
            losses['ent_c_reg'] = entropy_c_reg.sum()

            entropy_c_mixed = -torch.sum(mixed_c * torch.log(mixed_c),4)
            entropy_c_mixed = entropy_c_mixed.mul(tmasks)
            losses['entropy_c_mixed'] = entropy_c_mixed.sum()
        losses['entropy_srs'] = entropy_srs.sum()

    if 'policy_latent' in outputs:
        losses['re_observation'] = F.smooth_l1_loss(outputs['re_observation'], targets['re_observation'], reduction='none').mul(omasks).sum()
        b_size = outputs['re_policy'].shape[0]
        o_size = outputs['re_policy'].shape[1]
        a_size = outputs['re_policy'].shape[-1]
        o_re_p = torch.reshape(outputs['re_policy'], (b_size * o_size, a_size))
        t_re_p = torch.reshape(targets['re_policy'], (b_size * o_size, 1)).squeeze_()
        # Reconstruction loss
        losses['re_policy'] = torch.reshape(F.cross_entropy(o_re_p, t_re_p, reduction='none'), (b_size, o_size, 1, 1)).mul(omasks).sum()
        # Reconstruction policy entropy
        entropy_re_p = dist.Categorical(logits=outputs['re_policy']).entropy().mul(tmasks.mean(-1))
        c_args = args.get('contrastive_learning', {})
        # 生成モデルの loss 選択
        asc_type = args['agent'].get('ASC_type', False)
        if asc_type == 'VAE':
            # KL loss
            losses['p_l_KL'] = -0.5 * torch.sum(1 + outputs['log_dev'] - outputs['average']**2 - outputs['log_dev'].exp()) 
            # Contrastive_loss
            ## average
            half_average = torch.chunk(F.normalize(outputs['average'], dim=-1).mul(omasks), 2, dim=0)
            average_i = torch.reshape(half_average[0], (b_size * o_size, -1))
            average_j = torch.reshape(half_average[1], (b_size * o_size, -1))
            average_logits = torch.mm(average_i, average_j.t()) / c_args.get('temperature', 1.0)
            average_labels = torch.arange(average_logits.size(0))
            ## log_dev
            half_log_dev = torch.chunk(F.normalize(outputs['log_dev'], dim=-1).mul(omasks), 2, dim=0)
            log_dev_i = torch.reshape(half_log_dev[0], (b_size * o_size, -1))
            log_dev_j = torch.reshape(half_log_dev[1], (b_size * o_size, -1))
            log_dev_logits = torch.mm(log_dev_i, log_dev_j.t()) / c_args.get('temperature', 1.0)
            log_dev_labels = torch.arange(log_dev_logits.size(0))
            losses['contrast'] = (F.cross_entropy(average_logits, average_labels, reduction='none') + F.cross_entropy(log_dev_logits, log_dev_labels, reduction='none')).sum()
        elif asc_type == 'VQ-VAE':
            # coodbook loss
            # losses['vq_l_cb'] = F.smooth_l1_loss(outputs['quantized_policy_latent'], targets['policy_latent']).sum()
            losses['vq_l_cb'] = (((outputs['quantized_policy_latent'] - targets['policy_latent']) ** 2) / 2).sum()
            # commitment loss
            # losses['vq_l_cm'] = F.smooth_l1_loss(targets['quantized_policy_latent'], outputs['policy_latent']).sum()
            losses['vq_l_cm'] = (((targets['quantized_policy_latent'] - outputs['policy_latent']) ** 2) / 2).sum()
            losses['p_l_norm'] = torch.norm(outputs['policy_latent'], dim=-1).mean()
            losses['q_p_l_norm'] = torch.norm(outputs['quantized_policy_latent'], dim=-1).mean()
            # Contrastive_loss
            ## average
            half_latent = torch.chunk(F.normalize(outputs['policy_latent'], dim=-1).mul(omasks), 2, dim=0)
            latent_i = torch.reshape(half_latent[0], (b_size * o_size, -1))
            latent_j = torch.reshape(half_latent[1], (b_size * o_size, -1))
            latent_logits = torch.mm(latent_i, latent_j.t()) / c_args.get('temperature', 1.0)
            latent_labels = torch.arange(latent_logits.size(0))
            losses['contrast'] = F.cross_entropy(latent_logits, latent_labels, reduction='none').sum()

        # 生成モデル policy entropy を各種監視変数を追加
        losses['ent_re_p'] = entropy_re_p.sum()

    if 'policy_latent_set' in outputs:
        b_size = outputs['re_policy_set'].shape[0]
        s_size = outputs['re_policy_set'].shape[1]
        a_size = outputs['re_policy_set'].shape[-1]
        o_re_ps = torch.reshape(outputs['re_policy_set'], (b_size * s_size, a_size))
        t_re_ps = torch.reshape(targets['re_policy_set'], (b_size * s_size, 1)).squeeze_()
        # Reconstruction loss
        ## policy set
        losses['re_policy_set'] = torch.reshape(F.cross_entropy(o_re_ps, t_re_ps, reduction='none'), (b_size, s_size, 1, 1)).sum()
        ## observation set
        o_re_os = outputs['re_observation_set'].squeeze(-2)
        t_re_os = targets['re_observation_set'].squeeze(-2)
        ### 負の cos 類似度
        norm = torch.bmm(torch.norm(o_re_os, p=1, dim=2, keepdim=True), torch.norm(t_re_os, p=1, dim=2, keepdim=True).permute(0,2,1))
        dot = torch.bmm(o_re_os, t_re_os.permute(0,2,1))
        i_distances = -dot/norm
        w_i_distances = (torch.exp(i_distances)/torch.sum(torch.exp(i_distances), dim=-1, keepdim=True))
        mse = F.mse_loss(o_re_os.unsqueeze(-2).expand(-1, -1, i_distances.size(-1), -1), t_re_os.unsqueeze(-3).expand(-1, i_distances.size(-2), -1, -1), reduction='none')
        losses['re_observation_set'] = torch.mul(w_i_distances.unsqueeze(-1), mse).sum()
        # Reconstruction policy entropy
        entropy_re_ps = dist.Categorical(logits=o_re_ps).entropy().sum()
        # 生成モデルの loss 選択
        asc_type = args['agent'].get('ASC_type', False)
        c_args = args.get('contrastive_learning', {})
        if asc_type == 'SeTranVAE':
            # KL loss
            losses['p_l_KL_set'] = -0.5 * torch.sum(1 + outputs['log_dev_set'] - outputs['average_set'].pow(2) - outputs['log_dev_set'].exp()) 
            # Contrastive_loss
            ## average
            half_average = torch.chunk(F.normalize(outputs['average_set'], dim=-1), 2, dim=0)
            average_logits = torch.mm(half_average[0], half_average[1].t()) / c_args.get('temperature', 1.0)
            average_labels = torch.arange(average_logits.size(0))
            ## log_dev
            half_log_dev = torch.chunk(F.normalize(outputs['log_dev_set'], dim=-1), 2, dim=0)
            log_dev_logits = torch.mm(half_log_dev[0], half_log_dev[1].t()) / c_args.get('temperature', 1.0)
            log_dev_labels = torch.arange(log_dev_logits.size(0))
            losses['contrast_set'] = (F.cross_entropy(average_logits, average_labels) + F.cross_entropy(log_dev_logits, log_dev_labels))
        if asc_type == 'VQ-SeTranVAE':
            # coodbook loss
            # losses['vq_l_cb'] = F.smooth_l1_loss(outputs['quantized_policy_latent_set'], targets['policy_latent_set']).sum()
            losses['vq_l_cb'] = (((outputs['quantized_policy_latent_set'] - targets['policy_latent_set']) ** 2) / 2).sum()
            # commitment loss
            # losses['vq_l_cm'] = F.smooth_l1_loss(targets['quantized_policy_latent_set'], outputs['policy_latent_set']).sum()
            losses['vq_l_cm'] = (((targets['quantized_policy_latent_set'] - outputs['policy_latent_set']) ** 2) / 2).sum()
            losses['p_l_norm'] = torch.norm(outputs['policy_latent_set'], dim=-1).mean()
            losses['q_p_l_norm'] = torch.norm(outputs['quantized_policy_latent_set'], dim=-1).mean()
            # Contrastive_loss
            ## average
            half_latent = torch.chunk(F.normalize(outputs['policy_latent_set'], dim=-1), 2, dim=0)
            latent_logits = torch.mm(half_latent[0], half_latent[1].t()) / c_args.get('temperature', 1.0)
            latent_labels = torch.arange(latent_logits.size(0))
            losses['contrast_set'] = F.cross_entropy(latent_logits, latent_labels)

        losses['ent_re_ps'] = entropy_re_ps.sum()

    # エントロピー正則化のためのエントロピー計算
    entropy = dist.Categorical(logits=outputs['policy']).entropy().mul(tmasks.sum(-1))
    losses['ent'] = entropy.sum()

    factor = args.get('loss_factor', {})
    # value と policy の loss の計算
    base_loss = factor.get('rl', 1.0) * (losses['p'] + losses.get('v', 0) + losses.get('r', 0) + losses.get('q', 0) + losses.get('a', 0) + losses.get('c', 0)) + \
        factor.get('rnd', 1.0) * losses.get('rnd', 0) + \
        factor.get('recon', 1.0) * (losses.get('re_observation', 0) + losses.get('re_policy', 0)) +\
        factor.get('vae_kl', 1.0) * losses.get('p_l_KL', 0) +\
        factor.get('codebook', 1.0) * losses.get('vq_l_cb', 0) + \
        factor.get('commitment', 1.0) * losses.get('vq_l_cm', 0) +\
        factor.get('contrast', 1.0) * losses.get('contrast', 0) +\
        factor.get('recon_p_set', 1.0) * (losses.get('re_policy_set', 0) +\
        factor.get('recon_o_set', 1.0) *losses.get('re_observation_set', 0)) +\
        factor.get('vae_kl', 1.0) * losses.get('p_l_KL_set', 0) +\
        factor.get('contrast', 1.0) * losses.get('contrast_set', 0)
    # エントロピー正則化 loss の計算
    entropy_loss = entropy.mul(1 - batch['progress'] * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['total'] = base_loss + entropy_loss

    return losses, dcnt


def compute_loss(batch, model, target_model, metadataset, hidden, args):
    # target 計算に必要な forward 計算を行う
    outputs = forward_prediction(model, hidden, batch, args)
    # target-net の forward 計算を行う
    use_target_model = target_model is not None
    # TODO: hidden も target_model 用に用意
    if use_target_model:
        target_outputs = forward_prediction(target_model, hidden, batch, args)
    else:
        target_outputs = {}

    # learning_q = ('qvalues' in outputs)
    # burn_in_steps 分ずらす
    if args['burn_in_steps'] > 0:
        batch = map_r(batch, lambda v: v[:, args['burn_in_steps']:] if v.size(1) > 1 else v)
        outputs = map_r(outputs, lambda v: v[:, args['burn_in_steps']:])
    # action とを取り出す
    actions = batch['action']
    # episode masks (2 人対戦ゲームなどで自分 episode だけに限定するなど？) を取り出す
    emasks = batch['episode_mask']
    # vtrace で使う全体に対する係数（ハードコードで良いのか？）
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0
    # 挙動方策の確率を log したやつ 
    log_selected_b_policies = torch.log(torch.clamp(batch['selected_prob'], 1e-16, 1)) * emasks
    # 推定方策の確率を log したやつ 
    log_selected_t_policies = F.log_softmax(outputs['policy'], dim=-1).gather(-1, actions) * emasks

    # thresholds of importance sampling
    # log の引き算（上と合わせて方策確率同士の割り算を引き算に変換）
    log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
    # log を exp で確率比率に戻している（IS 確率比）
    rhos = torch.exp(log_rhos)
    # IS 確率比を v-trace と同様に ρ で clip している（状態ごと計算の事前に）
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    # IS 確率比を v-trace と同様に c で clip している（状態ごと計算の事前に）
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    # gradient が流れないように forward 計算結果を計算グラフから切り離している
    outputs_nograd = {k: o.detach() for k, o in outputs.items()}
    target_outputs_nograd = {k: o.detach() for k, o in target_outputs.items()}

    # compute targets and advantage
    targets = {}
    advantages = {}

    # if (('value' in outputs_nograd) and (use_target_model is not None)) or (('value' in target_outputs_nograd) and (use_target_model is None)):
    #     values_nograd = target_outputs_nograd['value'] if use_target_model else outputs_nograd['value']
    #     if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
    #         # 2 人対戦ゲームで相手から見た value なので負に符号反転している
    #         values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
    #         # 2 人対戦ゲームで自分と相手の value を計算して 2 で割るなどしている
    #         values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
    #     outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
            # 2 人対戦ゲームで相手から見た value なので負に符号反転している
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            # 2 人対戦ゲームで自分と相手の value を計算して 2 で割るなどしている
            values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)
    
    if 'qvalue' in outputs_nograd:
        # TODO: 対戦ゲームに未対応
        outputs['selected_qvalue'] = outputs['qvalue'].gather(-1, actions) * emasks
        if use_target_model:
            targets['qvalue'] = target_outputs_nograd['qvalue'] * emasks
        else:
            targets['qvalue'] = outputs_nograd['qvalue'] * emasks


    if 'embed_state' in outputs_nograd:
        targets['embed_state'] = outputs_nograd['embed_state_fix'].detach() * emasks
        outputs['loss_rnd'] = compute_rnd(outputs['embed_state'], targets['embed_state'])
        intrinsic_reward = outputs['loss_rnd'].detach().sum(-1, keepdim=True)
        if 'bonus' in batch:
            outputs['bonus'] = outputs['bonus'] + intrinsic_reward
        else:
            outputs['bonus'] = intrinsic_reward

    if 'confidence' in outputs:
        # 信頼度の学習
        targets['confidence'] = (actions * emasks).to(torch.long)

    if 'rl_latent' in outputs_nograd:
        # 学習はしないが latent を抜き出しておく
        targets['rl_latent'] = (outputs_nograd['rl_latent'] * emasks)

    if 'policy_latent' in outputs_nograd:
        targets['re_observation'] = batch['observation'].detach() * emasks
        targets['re_policy'] = (actions * emasks).to(torch.long).detach()
        targets['policy_latent'] = (outputs_nograd['policy_latent'] * emasks)
    if 'quantized_policy_latent' in outputs_nograd:
        targets['quantized_policy_latent'] = (outputs_nograd['quantized_policy_latent'] * emasks)

    if 'policy_latent_set' in outputs:
        targets['re_observation_set'] = batch['set']['observation'].detach()
        targets['re_policy_set'] = batch['set']['action'].to(torch.long).detach()
    if 'quantized_policy_latent_set' in outputs_nograd:
        targets['policy_latent_set'] = outputs_nograd['policy_latent_set']
        targets['quantized_policy_latent_set'] = outputs_nograd['quantized_policy_latent_set']

    # model forward 計算の value, 生の outcome, None, 終端フラグ, 適格度トレース値 λ, 割引率 γ=1, Vtorece で使う ρ, Vtorece で使う c 
    # value_args = outputs_nograd.get('value', None), batch['outcome'], None, args['lambda'], 1.0, clipped_rhos, cs
    # 割引率付き Return からの学習を定義（outcome と reward, return を統合したのでこれのみ）
    # model forward 計算の value, 割引率付き return, reward, 終端フラグ, 適格度トレース値 λ, 割引率 γ, Vtorece で使う ρ, Vtorece で使う c, bonus
    value_args = outputs_nograd.get('value', None), batch['return'], batch['reward'], batch['terminal'], args['lambda'], args['gamma'], clipped_rhos, cs, targets.get('qvalue', None), outputs.get('bonus', None)
    # model forward 計算の return, 生の return, 生の reward, 終端フラグ, 適格度トレース値 λ, 割引率 γ, Vtorece で使う ρ, Vtorece で使う c 
    # return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], args['lambda'], args['gamma'], clipped_rhos, cs

    # アルゴリズムに応じて target value を計算する
    results = compute_target(args['value_target'], *value_args)
    targets['value'], advantages['value'] = results['target_values'], results['advantages']
    if 'qvalue' in outputs_nograd:
        targets['advantage_for_q'] = results['target_advantage_for_q']
        targets['selected_qvalue'] = results['qvalue']

    # policy 用の advantage 計算が異なる場合，そのアルゴリズムに応じて target value を計算する
    if args['policy_target'] != args['value_target']:
        # _, advantages['value'] = compute_target(args['policy_target'], *value_args)
        # _, advantages['return'] = compute_target(args['policy_target'], *return_args)
        results = compute_target(args['policy_target'], *value_args)
        advantages['value'] = results['advantages']
        # results = compute_target(args['policy_target'], *return_args)
        # advantages['return'] = results['advantages']

    # compute policy advantage
    # IS 確率比を考慮して policy 更新に使う advantage の総計を計算する
    # value の advantages と return の advantages を足している
    # value は現在ほぼ outcome の直接学習
    # return は outputs に return がないと利用できていない
    total_advantages = clipped_rhos * sum(advantages.values())

    # ここまでは target value, advantage の計算を行うのがメイン
    # 具体的な loss 計算は compose_losses で行う
    return compose_losses(outputs, log_selected_t_policies, total_advantages, targets, batch, metadataset, args)


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.executor = MultiProcessJobExecutor(self._worker, self._selector(), self.args['num_batchers'])

    def _selector(self):
        while True:
            # batch_size 分 episode を切り出すジェネレータ
            yield [self.select_episode() for _ in range(self.args['batch_size'])]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while True:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args['burn_in_steps'], self.args['forward_steps'], self.args)
            if self.args['agent']['ASC_trajectory_length'] > 0:
                batch['set'] = make_batch(episodes, 0, self.args['agent']['ASC_trajectory_length'], self.args,
                        {'moment': 'moment_set', 'start':'start_set', 'end':'end_set', 'base':'base_set', 'train_start':'train_start_set'})
            conn.send(batch)
        print('finished batcher %d' % bid)

    def run(self):
        self.executor.start()
    
    def _rand_step(self, train_st, steps):
        st = max(0, train_st - self.args['burn_in_steps']) # burn_in_steps を考慮した開始 step 
        ed = min(train_st + self.args['forward_steps'], steps) # forward_steps を考慮した終了 step
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        return st, ed, st_block, ed_block

    def _rand_step_for_set(self, steps, trajectory_length):
        # A-S-C で集合として学習する場合に
        turn_candidates = 1 + max(0, steps - trajectory_length - 1 if self.args['return_buckup'] else 0)  # buckup 用 dummy 状態を除外 
        train_st_set = random.randrange(turn_candidates)
        st_set = train_st_set # 開始 step 
        ed_set = min(train_st_set + trajectory_length, steps) # 終了 step が極端に短い場合を考慮した終了 step
        st_block_set = st_set // self.args['compress_steps']
        ed_block_set = (ed_set - 1) // self.args['compress_steps'] + 1
        return train_st_set, st_set, ed_set, st_block_set, ed_block_set

    def select_episode(self):
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        train_st = random.randrange(turn_candidates)
        # st = max(0, train_st - self.args['burn_in_steps']) # burn_in_steps を考慮した開始 step 
        # ed = min(train_st + self.args['forward_steps'], ep['steps']) # forward_steps を考慮した終了 step
        # st_block = st // self.args['compress_steps']
        # ed_block = (ed - 1) // self.args['compress_steps'] + 1
        st, ed, st_block, ed_block = self._rand_step(train_st, ep['steps'])
        ep_minimum = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
        }
        if self.args['agent']['ASC_trajectory_length'] > 0:
            train_st_set, st_set, ed_set, st_block_set, ed_block_set = self._rand_step_for_set(ep['steps'], self.args['agent']['ASC_trajectory_length'])
            ep_minimum = {**ep_minimum, **{
                'moment_set': ep['moment'][st_block_set:ed_block_set],
                'base_set': st_block_set * self.args['compress_steps'],
                'start_set': st_set, 'end_set': ed_set, 'train_start_set': train_st_set,
            }}
        ep_contrast = None
        if self.args['contrastive_learning']['use']:
            contrast_st = random.randrange(turn_candidates)
            c_st, c_ed, c_st_block, c_ed_block = self._rand_step(contrast_st, ep['steps'])
            ep_contrast = {
                'args': ep['args'], 'outcome': ep['outcome'],
                'moment': ep['moment'][c_st_block:c_ed_block],
                'base': c_st_block * self.args['compress_steps'],
                'start': c_st, 'end': c_ed, 'train_start': contrast_st, 'total': ep['steps'],
            }
            if self.args['agent']['ASC_trajectory_length'] > 0:
                contrast_st_set, c_st_set, c_ed_set, c_st_block_set, c_ed_block_set = self._rand_step_for_set(ep['steps'], self.args['agent']['ASC_trajectory_length'])
                ep_contrast = {**ep_contrast, **{
                    'moment_set': ep['moment'][c_st_block_set:c_ed_block_set],
                    'base_set': c_st_block_set * self.args['compress_steps'],
                    'start_set': c_st_set, 'end_set': c_ed_set, 'train_start_set': contrast_st_set,
                }}
            return [ep_minimum, ep_contrast]
            # ep_minimum = {**ep_minimum, **ep_contrast}
        #     ep['outcome'] = torch.cat((ep['outcome'], ep['outcome']), 0)
        #     ep['moment'][st_block:ed_block] = torch.cat((ep['moment'][st_block:ed_block], ep['moment'][c_st_block:c_ed_block]), 0)
        #     st_block = torch.cat((st_block, c_st_block), 0)
        #     st = torch.cat((st, c_st), 0)
        #     ed = torch.cat((ed, c_ed), 0), 
        #     train_st = torch.cat((train_st, contrast_st), 0), 
        #     ep['steps'] = torch.cat((ep['steps'], ep['steps']), 0),
        # ep_minimum = {
        #     'args': ep['args'], 'outcome': ep['outcome'],
        #     'moment': ep['moment'][st_block:ed_block],
        #     'base': st_block * self.args['compress_steps'],
        #     'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
        # }
        return [ep_minimum]

    def batch(self):
        return self.executor.recv()


class Trainer:
    def __init__(self, args, model, target_model, metadataset):
        self.episodes = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.metadataset = metadataset
        self.default_lr = self.args['default_learning_rate'] # 3.0e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.default_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        self.batcher = Batcher(self.args, self.episodes)
        self.update_flag = False
        self.update_queue = queue.Queue(maxsize=1)
        # TODO: Twin Delayed の実装方法を検討
        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model
        if self.gpu > 1:
            self.trained_model = nn.DataParallel(self.wrapped_model)
        
        if target_model is not None:
            self.t_model = target_model
            self.wrapped_t_model = ModelWrapper(self.t_model)
            self.target_model = self.wrapped_t_model
            if self.gpu > 1:
                self.target_model = nn.DataParallel(self.wrapped_t_model)
            self.target_update = self.args['target_model'].get('update', None)
            self.target_param = self.args['target_model'].get('update_param', None)
        else:
            self.target_model = None
            self.wrapped_t_model = None
            self.target_model = None
            self.target_update = None
            self.target_param = None

    def update(self):
        self.update_flag = True
        model, steps = self.update_queue.get()
        return model, steps

    def train(self):
        if self.optimizer is None:  # non-parametric model
            time.sleep(0.1)
            return self.model

        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        if self.gpu > 0:
            self.trained_model.cuda()
        self.trained_model.train()

        # モデルの学習
        while data_cnt == 0 or not self.update_flag:
            batch = self.batcher.batch()
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = self.wrapped_model.init_hidden([batch_size, player_count])
            if self.gpu > 0:
                batch = to_gpu(batch)
                hidden = to_gpu(hidden)

            # loss の計算
            losses, dcnt = compute_loss(batch, self.trained_model, self.target_model, self.metadataset, hidden, self.args)
            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            # target net の更新
            if self.target_update == 'soft':
                tau = self.target_param
                with torch.no_grad():
                    for target_param, trained_param in zip(self.t_model.parameters(), self.model.parameters()):
                        target_param.data.copy_(tau * trained_param.data + (1.0- tau) * target_param.data)
            elif self.target_update == 'hard':
                interval_episodes = self.target_param
                if self.steps % interval_episodes == 0:
                    with torch.no_grad():
                        target_param.data.copy_(trained_param.data)

            batch_cnt += 1
            data_cnt += dcnt
            for k, l in losses.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            self.steps += 1

        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.default_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        return copy.deepcopy(self.model)

    def run(self):
        print('waiting training')
        # minimum_episode
        while len(self.episodes) < self.args['minimum_episodes']:
            time.sleep(1)
        if self.optimizer is not None:
            self.batcher.run()
            print('started training')
        while True:
            model = self.train()
            self.update_flag = False
            self.update_queue.put((model, self.steps))
        print('finished training')


class Learner:
    def __init__(self, args, net=None, remote=False):
        train_args = args['train_args']
        env_args = args['env_args']
        train_args['env'] = env_args
        args = train_args

        self.args = args
        random.seed(args['seed'])

        self.env = make_env(env_args)
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.shutdown_flag = False
        self.flags = set()
        self.saving_interval_epochs = args['saving_interval_epochs']

        # trained datum
        self.model_epoch = self.args['restart_epoch']
        self.model = net if net is not None else self.env.net(args['agent'])
        if self.model_epoch > 0:
            self.model.load_state_dict(torch.load(self.model_path(self.model_epoch)), strict=False)
        # used target_model
        if args['target_model'].get('use', False):
            self.target_model = net if net is not None else self.env.net(args['agent'])
            if self.model_epoch > 0:
                self.target_model.load_state_dict(torch.load(self.model_path(self.model_epoch)), strict=False)
        else: 
            self.target_model = None

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0
        self.num_returned_episodes = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0

        # KNN
        self.metadataset = {}
        if 'knn' in args['metadata']['name']:
            self.metadataset['knn'] = KNN(args)
        if 'global_aleph' in args['metadata']['name']:
            self.metadataset['global_aleph'] = args['metadata']['global_aleph']
        if 'regional_weight' in args['metadata']['name']:
            self.metadataset['regional_weight'] = args['metadata']['regional_weight']

        # multiprocess or remote connection
        self.worker = WorkerServer(args) if remote else WorkerCluster(args)

        # thread connection
        self.trainer = Trainer(args, self.model, self.target_model, self.metadataset)

        # episode count
        self.uns_bool = env_args['param']['uns_setting']['uns_bool'] # 非定常のフラグ
        self.uns_num = env_args['param']['uns_setting']['uns_num'] # 非定常の周期
        #print(self.uns_bool)
        #print(self.uns_num)

        # time count
        self.time_start = 0
        self.time_end = 0

    def model_path(self, model_id):
        return os.path.join('models', str(model_id) + '.pth')

    def latest_model_path(self):
        return os.path.join('models', 'latest.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_epoch += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        if self.model_epoch % self.saving_interval_epochs == 0 or self.saving_interval_epochs <= 0:
            torch.save(model.state_dict(), self.model_path(self.model_epoch))
            torch.save(model.state_dict(), self.latest_model_path())

    def feed_episodes(self, episodes):
        # analyze generated episodes
        for episode in episodes:
            #print("learner_episode: ", self.num_returned_episodes)
            if episode is None:
                continue
            for p in episode['args']['player']:
                model_id = episode['args']['model_id'][p]
                outcome = episode['outcome'][p]
                n, r, r2 = self.generation_results.get(model_id, (0, 0, 0))
                self.generation_results[model_id] = n + 1, r + outcome, r2 + outcome ** 2
            self.num_returned_episodes += 1
            if self.num_returned_episodes % 50 == 0:
                print(self.num_returned_episodes, end=' ', flush=True)
            if self.uns_bool:
                if self.num_returned_episodes % self.uns_num == 0:
                    print("learner_uns : ")
                    self.env.uns()

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])

        mem_percent = psutil.virtual_memory().percent
        mem_ok = mem_percent <= 95
        # maxmum_episode のサイズ制限がおかしい
        # maximum_episodes = self.args['maximum_episodes'] if mem_ok else int(len(self.trainer.episodes) * 95 / mem_percent)
        maximum_episodes = self.args['maximum_episodes']

        if not mem_ok and 'memory_over' not in self.flags:
            warnings.warn("memory usage %.1f%% with buffer size %d" % (mem_percent, len(self.trainer.episodes)))
            self.flags.add('memory_over')

        # エピソードが一定以上溜まったら吐き出す
        while len(self.trainer.episodes) > maximum_episodes:
            # 溢れた episode の pop
            self.trainer.episodes.popleft()

    def feed_results(self, results):
        # store evaluation results
        for result in results:
            if result is None:
                continue
            for p in result['args']['player']:
                model_id = result['args']['model_id'][p]
                res = result['result'][p]
                n, r, r2 = self.results.get(model_id, (0, 0, 0))
                # self.results[model_id] = n + 1, r + res, r2 + res ** 2
                self.results[model_id] = n + 1, r + res, r2 + abs(res)

                if model_id not in self.results_per_opponent:
                    self.results_per_opponent[model_id] = {}
                opponent = result['opponent']
                n, r, r2 = self.results_per_opponent[model_id].get(opponent, (0, 0, 0))
                self.results_per_opponent[model_id][opponent] = n + 1, r + res, r2 + res ** 2

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_epoch)

        if self.model_epoch not in self.results:
            print('win rate = Nan (0)')
        else:
            def output_wp(name, results):
                n, r, r2 = results
                mean = r / (n + 1e-6)
                name_tag = ' (%s)' % name if name != '' else ''
                print('win rate%s = %.3f (%.1f / %d)' % (name_tag, (mean + 1) / 2, (r + n) / 2, n))
                print('average reward%s = %.3f (%.1f / %d)' % (name_tag, mean, (r + n) / 2, n))

            keys = self.results_per_opponent[self.model_epoch]
            if len(self.args.get('eval', {}).get('opponent', [])) <= 1 and len(keys) <= 1:
                output_wp('', self.results[self.model_epoch])
            else:
                output_wp('total', self.results[self.model_epoch])
                for key in sorted(list(self.results_per_opponent[self.model_epoch])):
                    output_wp(key, self.results_per_opponent[self.model_epoch][key])

        if self.model_epoch not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_epoch]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

        model, steps = self.trainer.update()
        if model is None:
            model = self.model
        self.update_model(model, steps)

        # clear flags
        self.flags = set()

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        self.time_start = time.perf_counter()
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        # no update call before storing minimum number of episodes + 1 epoch
        next_update_episodes = prev_update_episodes + self.args['update_episodes']

        while self.worker.connection_count() > 0 or not self.shutdown_flag:
            # req は worker から受けるリクエストメッセージで処理の切り替えに使う
            # data はリクエストメッセージの付加情報
            # send_data は worker に送り返すデータ
            try:
                conn, (req, data) = self.worker.recv(timeout=0.3)
            except queue.Empty:
                continue

            multi_req = isinstance(data, list)
            if not multi_req:
                data = [data]
            send_data = []

            if req == 'args':
                if self.shutdown_flag:
                    send_data = [None] * len(data)
                else:
                    for _ in data:
                        args = {'model_id': {}, 'metadata_id': {}}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        else:
                            args['role'] = 'g'

                        if args['role'] == 'g':
                            # genatation configuration
                            args['player'] = self.env.players()
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                    args['metadata_id'][p] = self.model_epoch
                                    # args['metadata_id'][p] = self.num_episodes
                                else:
                                    args['model_id'][p] = -1
                                    args['metadata_id'][p] = -1
                            self.num_episodes += 1

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = [self.env.players()[self.num_results % len(self.env.players())]]
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                    args['metadata_id'][p] = self.model_epoch
                                    # args['metadata_id'][p] = self.num_episodes
                                else:
                                    args['model_id'][p] = -1
                                    args['metadata_id'][p] = -1
                            self.num_results += 1

                        send_data.append(args)

            elif req == 'episode':
                # report generated episodes
                self.feed_episodes(data)
                send_data = [None] * len(data)

            elif req == 'result':
                # report evaluation results
                self.feed_results(data)
                send_data = [None] * len(data)

            elif req == 'return_metadata':
                # report evaluation results
                if 'knn' in self.args['metadata']['name']:
                    # feed_knn(self.knn, self.args, data)
                    feed_knn(self.metadataset['knn'], self.args, data)
                send_data = [None] * len(data)

            elif req == 'model':
                for model_id in data:
                    model = self.model
                    if model_id != self.model_epoch and model_id > 0:
                        try:
                            model = copy.deepcopy(self.model)
                            model.load_state_dict(torch.load(self.model_path(model_id)), strict=False)
                        except:
                            # return latest model if failed to load specified model
                            pass
                    send_data.append(pickle.dumps(model))

            elif req == 'metadata':
                # trainer に保存して欲しい情報全般を取り出すリクエストメッセージ
                self.metadataset['num_episodes'] = self.num_episodes
                send_data.append(pickle.dumps(self.metadataset))

            if not multi_req and len(send_data) == 1:
                send_data = send_data[0]
            self.worker.send(conn, send_data)

            if self.num_returned_episodes >= next_update_episodes:
                prev_update_episodes = next_update_episodes
                next_update_episodes = prev_update_episodes + self.args['update_episodes']
                self.update()
                if self.args['epochs'] >= 0 and self.model_epoch >= self.args['epochs']:
                    self.shutdown_flag = True
        print('finished server')
        self.time_end = time.perf_counter()
        print('time :',self.time_end - self.time_start)

    def run(self):
        # open training thread
        threading.Thread(target=self.trainer.run, daemon=True).start()
        # open generator, evaluator
        self.worker.run()
        self.server()


def train_main(args):
    prepare_env(args['env_args'])  # preparing environment is needed in stand-alone mode
    learner = Learner(args=args)
    learner.run()


def train_server_main(args):
    learner = Learner(args=args, remote=True)
    learner.run()
