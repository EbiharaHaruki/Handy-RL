# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of simpletask

import copy
import random
from matplotlib.pyplot import axis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import sys
import datetime

from torch.nn import init
import torch.optim as optim

from ..environment import BaseEnvironment


class SimpleModel(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n)
        self.head_v = nn.Linear(nn_size, 1)

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h = F.relu(self.fc1(o))
        p = self.head_p(h)
        v = self.head_v(h)
        # return {'policy': p, 'value': torch.tanh(v)}
        return {'policy': p, 'value': v}


# Q-Learning
class PVQModel(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, 2**hyperplane_n) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h_l = self.fc1(o)
        h = F.relu(h_l)
        p = self.head_p(h)
        v = self.head_v(h)
        a = self.head_a(h)
        b = self.head_b(h)
        q = b + a - a.sum(-1).unsqueeze(-1)
        return {
            'policy': p, 'value': v, 
            'advantage_for_q': a, 'qvalue': q, 'rl_latent': h_l}


# RSRS
class PVQCModel(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, 2**hyperplane_n) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        self.head_c = nn.Linear(nn_size, 2**hyperplane_n) # 信頼度(confidence rate)

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h_l = self.fc1(o)
        h = F.relu(h_l)
        p = self.head_p(h)
        v = self.head_v(h)
        a = self.head_a(h)
        b = self.head_b(h)
        q = b + a - a.sum(-1).unsqueeze(-1)
        c = self.head_c(h)
        return {
            'policy': p, 'value': v, 
            'advantage_for_q': a, 'qvalue': q, 'rl_latent': h_l, 'confidence': c}


# RND
class RNDModel(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        # RND head dim：探索の複雑性
        rnd_head_size = 16
        ## 学習
        self.fc_rnd = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_rnd = nn.Linear(nn_size, rnd_head_size)
        ## 固定
        self.fc_rnd_fix = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_rnd_fix = nn.Linear(nn_size, rnd_head_size)

    def forward(self, o_in, hidden=None):
        rnd = F.relu(self.fc_rnd(o_in))
        rnd = self.head_rnd(rnd)
        rnd_fix = F.relu(self.fc_rnd_fix(o_in))
        rnd_fix = self.head_rnd_fix(rnd_fix)
        return {'embed_state': rnd, 'embed_state_fix': rnd_fix}


# RL with RND wrapper model
class RLwithRNDModel(nn.Module):
    def __init__(self, args, hyperplane_n, rl_net):
        super().__init__()
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.rnd_net = RNDModel(args, hyperplane_n)

    def forward(self, x, hidden=None):
        o_in = x.get('o', None)
        # RL model
        out = self.rl_net(x)
        # rnd
        out_rnd = self.rnd_net(o_in)
        out = {**out, **out_rnd}
        return out


# VAE パラメータ
vae_latent_dim = 32
vae_nn_size = [258, 128]

# VAE
## Encoder
class Encoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.action_num = 2**hyperplane_n
        self.fc1 = nn.Linear(hyperplane_n + 1 + self.action_num, vae_nn_size[0])
        self.fc2 = nn.Linear(vae_nn_size[0], vae_nn_size[1])
        self.fc_ave = nn.Linear(vae_nn_size[1], vae_latent_dim) # 平均
        self.fc_dev = nn.Linear(vae_nn_size[1], vae_latent_dim) # 標準偏差
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, o_in, a_in, hidden=None):
        a_hot = F.one_hot(torch.squeeze(a_in), num_classes=self.action_num)
        x = torch.cat((o_in, a_hot), 1)
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        ave = self.fc_ave(h)
        log_dev = self.fc_dev(h)

        # Reparametrization Trick
        epsilon = torch.randn_like(ave)  # 平均0分散1の正規分布に従い生成されるz_dim次元の乱数ε
        policy_latent = ave + torch.exp(log_dev / 2) * epsilon  # ζ = μ + σε
        return {'policy_latent': policy_latent, 'average': ave, 'log_dev': log_dev}

## Action decoder
class ActionDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(vae_latent_dim + hyperplane_n + 1, vae_nn_size[1])
        self.fc2 = nn.Linear(vae_nn_size[1], vae_nn_size[0])
        self.fc_p = nn.Linear(vae_nn_size[0], 2**hyperplane_n)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, latent, o_in, hidden=None):
        x = torch.cat((latent, o_in), 1)
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_p = self.fc_p(h)
        return {'re_policy': re_p}

## Observation decoder
class ObservationDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(vae_latent_dim, vae_nn_size[1])
        self.fc2 = nn.Linear(vae_nn_size[1], vae_nn_size[0])
        self.fc_o = nn.Linear(vae_nn_size[0], hyperplane_n + 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, latent, hidden=None):
        h = self.fc1(latent)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_o = self.fc_o(h)
        return {'re_observation': re_o}

## Action-Observation VAE
class AOVAE(nn.Module):
    def __init__(self, args, encoder, a_decoder, o_decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.a_decoder = a_decoder
        self.o_decoder = o_decoder

    def forward(self, o_in, a_in, hidden=None):
        h_l = self.encoder(o_in, a_in)
        re_p = self.a_decoder(h_l['policy_latent'], o_in)
        re_s = self.o_decoder(h_l['policy_latent'])
        return {**re_p, **re_s, **h_l}

# RL with VAE wrapper model (ASC)
class ASCModel(nn.Module):
    def __init__(self, args, hyperplane_n, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = Encoder(args, hyperplane_n)
        self.action_decoder = ActionDecoder(args, hyperplane_n)
        self.observation_decoder = ObservationDecoder(args, hyperplane_n)
        self.vae = AOVAE(args, self.encoder, self.action_decoder, self.observation_decoder)

    def forward(self, x, hidden=None):
        o_in = x.get('o', None)
        a_in = x.get('a', None)
        latent = x.get('latent', None)
        # RL model
        out = self.rl_net(x)
        # State-Action VAE
        out_g = self.vae(o_in, a_in) if a_in!=None else None
        if out_g is not None:
            out = {**out, **out_g}
        # VAE action decoder
        out_g_p = self.action_decoder(o_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        return out

# VQ-VAE パラメータ
vq_latent_size= 4 # lattent の個数，lattent の次元数そのものは vq_latent_dim * vq_embedding_dim
vq_embedding_size = 8 # codebook の code　(embedding vector) の数
vq_embedding_dim = 4 # embedding vector 1 つずつの長さ
vq_nn_size = [258, 128]

# VQ-VAE
## Encoder (VQ)
class VQEncoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.action_num = 2**hyperplane_n
        self.fc1 = nn.Linear(hyperplane_n + 1 + self.action_num, vae_nn_size[0])
        self.fc2 = nn.Linear(vae_nn_size[0], vae_nn_size[1])
        self.fc_latent = nn.Linear(vae_nn_size[1], vq_latent_size * vq_embedding_dim) # 潜在変数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, o_in, a_in, hidden=None):
        a_hot = F.one_hot(torch.squeeze(a_in), num_classes=self.action_num)
        x = torch.cat((o_in, a_hot), 1)
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        policy_latent = self.fc_latent(h)
        # policy_latent = policy_latent.view((policy_latent.size(0), vq_latent_dim, vq_embedding_dim))
        return {'policy_latent': policy_latent}

## Action decoder (VQ)
class ActionVQDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(vq_latent_size * vq_embedding_dim + hyperplane_n + 1, vae_nn_size[1])
        self.fc2 = nn.Linear(vae_nn_size[1], vae_nn_size[0])
        self.fc_p = nn.Linear(vae_nn_size[0], 2**hyperplane_n)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, latent, o_in, hidden=None):
        # latent_flattened = latent.view((latent.size(0), vq_latent_dim * vq_embedding_dim))
        latent_flattened = latent
        x = torch.cat((latent_flattened, o_in), 1)
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_p = self.fc_p(h)
        return {'re_policy': re_p}

## Observation decoder (VQ)
class ObservationVQDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(vq_latent_size * vq_embedding_dim, vae_nn_size[1])
        self.fc2 = nn.Linear(vae_nn_size[1], vae_nn_size[0])
        self.fc_o = nn.Linear(vae_nn_size[0], hyperplane_n + 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, latent, hidden=None):
        # latent_flattened = latent.view((latent.size(0), vq_latent_dim * vq_embedding_dim))
        latent_flattened = latent
        h = self.fc1(latent_flattened)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_o = self.fc_o(h)
        return {'re_observation': re_o}

## Vector quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.latent_size = vq_latent_size
        self.embedding_size = vq_embedding_size
        self.embedding_dim = vq_embedding_dim
        self.codebook = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.embedding_size, 1/self.embedding_size)

    def forward(self, latent):
        latent_flattened = latent.contiguous().view(-1, self.embedding_dim) # [batch_size * forward_steps * player_num * latent_dim, embedding_dim]
        distances = (torch.sum(latent_flattened**2, dim=-1, keepdim=True)
                     + torch.sum(self.codebook.weight**2, dim=-1)
                     - 2 * torch.matmul(latent_flattened, self.codebook.weight.t()))

        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.embedding_size, device=latent.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_latent = torch.matmul(encodings, self.codebook.weight).view(latent.size())
        policy_vq_latent = latent + (quantized_latent - latent).detach()
        return {'policy_vq_latent': policy_vq_latent, 'quantized_policy_latent': quantized_latent}

## Action-Observation VQ-VAE
class AOVQVAE(nn.Module):
    def __init__(self, args, encoder, a_decoder, o_decoder, quantizer):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.quantizer = quantizer
        self.a_decoder = a_decoder
        self.o_decoder = o_decoder

    def forward(self, o_in, a_in, hidden=None):
        h_l = self.encoder(o_in, a_in)
        re_q = self.quantizer(h_l['policy_latent'])
        re_p = self.a_decoder(re_q['policy_vq_latent'], o_in)
        re_s = self.o_decoder(re_q['policy_vq_latent'])
        return {**re_p, **re_s, **h_l, **re_q}

# RL with VAE wrapper model (ASC)
class ASCVQModel(nn.Module):
    def __init__(self, args, hyperplane_n, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = VQEncoder(args, hyperplane_n)
        self.quantizer = VectorQuantizer(args)
        self.action_decoder = ActionVQDecoder(args, hyperplane_n)
        self.observation_decoder = ObservationVQDecoder(args, hyperplane_n)
        self.vqvae = AOVQVAE(args, self.encoder, self.action_decoder, self.observation_decoder, self.quantizer)

    def forward(self, x, hidden=None):
        o_in = x.get('o', None)
        a_in = x.get('a', None)
        latent = x.get('latent', None)
        # RL model
        out = self.rl_net(x)
        # State-Action VQ-VAE
        out_g = self.vqvae(o_in, a_in) if a_in!=None else None
        if out_g is not None:
            out = {**out, **out_g}
        # VQ-VAE action decoder
        out_g_p = self.action_decoder(o_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        return out


# VAE + Transformer (SeTranVAE) パラメータ 
tvae_latent_dim = 32 # 潜在変数の次元数
tvae_emb_size = 64 # transformer に入力する潜在変数を線形変換したサイズ
tvae_ffn_size = 256 # transformer に FFN 中間ユニット数
tvae_cnn_size = [128, 256, 128, 64] # action decoder の中間ユニット数 
tvae_head_num = 2 # ヘッド数
tvae_tf_layer_num = 3 # transformer のレイヤー数
tvae_noise_num = 9 # observation decoder の target 入力するノイズ数

#  VAE + Transformer
## Encoder
class TranEncoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.action_num = 2**hyperplane_n
        self.feature_num = hyperplane_n + 1 + self.action_num
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(self.feature_num), out_channels=tvae_emb_size, kernel_size=1)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(tvae_emb_size, tvae_head_num, tvae_ffn_size, dropout=self.args['ASC_dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, tvae_tf_layer_num)
        self.fc_ave = nn.Linear(tvae_emb_size, tvae_latent_dim) # 平均
        self.fc_dev = nn.Linear(tvae_emb_size, tvae_latent_dim) # 標準偏差
        self.bn = nn.BatchNorm1d(tvae_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, os_in, as_in, oa_mask=None, hidden=None):
        as_hot = F.one_hot(torch.squeeze(as_in), num_classes=self.action_num)
        x = torch.cat((os_in, as_hot), -1)
        h = torch.permute(x, (0, 2, 1))
        h = self.conv(h) 
        h = torch.permute(h, (0, 2, 1))
        h = self.transformer_encoder(h).mean(dim=-2)
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        ave = self.fc_ave(h)
        log_dev = self.fc_dev(h)
        return {'average_set': ave, 'log_dev_set': log_dev}

## Action decoder
class ActionConvDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=(tvae_latent_dim + hyperplane_n + 1), out_channels=tvae_cnn_size[0], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=tvae_cnn_size[0], out_channels=tvae_cnn_size[1], kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=tvae_cnn_size[1], out_channels=tvae_cnn_size[2], kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=tvae_cnn_size[2], out_channels=tvae_cnn_size[3], kernel_size=1)
        self.conv_p = nn.Conv1d(in_channels=tvae_cnn_size[3], out_channels=2**hyperplane_n, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(tvae_cnn_size[0])
        self.bn2 = nn.BatchNorm1d(tvae_cnn_size[1])
        self.bn3 = nn.BatchNorm1d(tvae_cnn_size[2])
        self.bn4 = nn.BatchNorm1d(tvae_cnn_size[3])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, latent, os_in, hidden=None):
        x = torch.cat((latent, os_in), -1)
        h = torch.permute(x, (0, 2, 1))
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_ps = self.conv_p(h)
        re_ps = torch.permute(re_ps, (0, 2, 1))
        return {'re_policy_set': re_ps}

## Observation decoder
class ObservationTranDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(tvae_latent_dim), out_channels=tvae_emb_size, kernel_size=1)
        self.conv_o= nn.Conv1d(in_channels=tvae_emb_size, out_channels=hyperplane_n + 1, kernel_size=1)
        # Transformer Encoder
        decoder_layer = nn.TransformerDecoderLayer(tvae_emb_size, tvae_head_num, tvae_ffn_size, dropout=self.args['ASC_dropout'], batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, tvae_tf_layer_num)
        self.bn = nn.BatchNorm1d(tvae_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, noise, latent, hidden=None):
        # noise linear transformation
        h_n = torch.permute(noise, (0, 2, 1))
        h_n = self.conv(h_n) 
        h_n = torch.permute(h_n, (0, 2, 1))
        # latent linear transformation
        h_l = torch.permute(latent, (0, 2, 1))
        h_l = self.conv(h_l) 
        h_l = torch.permute(h_l, (0, 2, 1))
        # transformer decoder 
        h = self.transformer_decoder(h_n, h_l)
        h = torch.permute(h, (0, 2, 1))
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_os = self.conv_o(h)
        re_os = torch.permute(re_os, (0, 2, 1))
        return {'re_observation_set': re_os}

## Action-Observation VAE
class AOTranVAE(nn.Module):
    def __init__(self, args, encoder, a_decoder, o_decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.a_decoder = a_decoder
        self.o_decoder = o_decoder

    def forward(self, os_in, as_in, hidden=None):
        h_l = self.encoder(os_in, as_in)
        average = h_l['average_set'].unsqueeze(1)
        log_dev = h_l['log_dev_set'].unsqueeze(1)

        # Latent generation and Reparametrization Trick
        epsilon = torch.randn_like(average.expand(-1, os_in.size(1), -1))  # 乱数ε
        policy_latent_set = average + torch.exp(log_dev / 2) * epsilon  # ζ = μ + σε
        # Noise generation and Reparametrization Trick (detach)
        epsilon_noise = torch.randn_like(average.expand(-1, tvae_noise_num, -1))  # 乱数ε
        noise_latent_set = (average + torch.exp(log_dev / 2) * epsilon_noise).detach()  # ζ = μ + σε

        re_p = self.a_decoder(policy_latent_set, os_in)
        re_s = self.o_decoder(noise_latent_set, policy_latent_set)
        return {**re_p, **re_s, **h_l, 'policy_latent_set': policy_latent_set}

# RL with VAE wrapper model (ASC)
class TranASCModel(nn.Module):
    def __init__(self, args, hyperplane_n, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = TranEncoder(args, hyperplane_n)
        self.action_decoder = ActionConvDecoder(args, hyperplane_n)
        self.observation_decoder = ObservationTranDecoder(args, hyperplane_n)
        self.vae = AOTranVAE(args, self.encoder, self.action_decoder, self.observation_decoder)

    def forward(self, x, hidden=None):
        os_in = x.get('os', None)
        as_in = x.get('as', None)
        latent = x.get('latent', None)
        # RL model
        out = self.rl_net(x)
        # State-Action VAE
        out_g = self.vae(os_in, as_in) if as_in!=None else None
        if out_g is not None:
            out = {**out, **out_g}
        # VAE action decoder
        out_g_p = self.action_decoder(os_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        return out


#  VQ-VAE + Transformer (SeTranVQ-VAE) パラメータ
# VQ-VAE パラメータ
tvq_latent_size= 4 # lattent の個数，lattent の次元数そのものは vq_latent_dim * vq_embedding_dim
tvq_embedding_size = 8 # codebook の code　(embedding vector) の数
tvq_embedding_dim = 8 # embedding vector 1 つずつの長さ

tvq_emb_size = 64 # transformer に入力する潜在変数を線形変換したサイズ
tvq_ffn_size = 256 # transformer に FFN 中間ユニット数
tvq_cnn_size = [128, 256, 128, 64] # action decoder の中間ユニット数 
tvq_head_num = 2 # ヘッド数
tvq_tf_layer_num = 3 # transformer のレイヤー数
tvq_noise_num = 9 # observation decoder の target 入力するノイズ数

#  VAE + Transformer
## Encoder
class TranVQEncoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        self.action_num = 2**hyperplane_n
        self.feature_num = hyperplane_n + 1 + self.action_num
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(self.feature_num), out_channels=tvq_emb_size, kernel_size=1)
        # # Positional Encoding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(tvq_emb_size, tvq_head_num, tvq_ffn_size, dropout=self.args['ASC_dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, tvq_tf_layer_num)
        self.fc_latent = nn.Linear(tvq_emb_size, tvq_latent_size * tvq_embedding_dim) # 生の潜在変数
        self.bn = nn.BatchNorm1d(tvq_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, os_in, as_in, oa_mask=None, hidden=None):
        as_hot = F.one_hot(torch.squeeze(as_in), num_classes=self.action_num)
        x = torch.cat((os_in, as_hot), -1)
        h = torch.permute(x, (0, 2, 1))
        h = self.conv(h) 
        h = torch.permute(h, (0, 2, 1))
        h = self.transformer_encoder(h).mean(dim=-2)
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        latent = self.fc_latent(h)
        return {'policy_latent_set': latent}

## Action decoder
class ActionConvVQDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=(tvq_latent_size * tvq_embedding_dim + hyperplane_n + 1), out_channels=tvq_cnn_size[0], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=tvq_cnn_size[0], out_channels=tvq_cnn_size[1], kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=tvq_cnn_size[1], out_channels=tvq_cnn_size[2], kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=tvq_cnn_size[2], out_channels=tvq_cnn_size[3], kernel_size=1)
        self.conv_p = nn.Conv1d(in_channels=tvq_cnn_size[3], out_channels=2**hyperplane_n, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(tvq_cnn_size[0])
        self.bn2 = nn.BatchNorm1d(tvq_cnn_size[1])
        self.bn3 = nn.BatchNorm1d(tvq_cnn_size[2])
        self.bn4 = nn.BatchNorm1d(tvq_cnn_size[3])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, vq_latent, os_in, hidden=None):
        # latents = latent.unsqueeze(1).expand(-1, os_in.size(1), -1)
        x = torch.cat((vq_latent, os_in), -1)
        h = torch.permute(x, (0, 2, 1))
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_ps = self.conv_p(h)
        re_ps = torch.permute(re_ps, (0, 2, 1))
        return {'re_policy_set': re_ps}

## Observation decoder
class ObservationTranVQDecoder(nn.Module):
    def __init__(self, args, hyperplane_n):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(tvq_embedding_dim), out_channels=tvq_emb_size, kernel_size=1)
        self.conv_o= nn.Conv1d(in_channels=tvq_emb_size, out_channels=hyperplane_n + 1, kernel_size=1)
        # # Positional Encoding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        # Transformer Encoder
        decoder_layer = nn.TransformerDecoderLayer(tvq_emb_size, tvq_head_num, tvq_ffn_size, dropout=self.args['ASC_dropout'], batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, tvq_tf_layer_num)
        self.bn = nn.BatchNorm1d(tvq_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])

    def forward(self, noise, vq_latent, hidden=None):
        # noise linear transformation
        h_n = torch.permute(noise, (0, 2, 1))
        h_n = self.conv(h_n) 
        h_n = torch.permute(h_n, (0, 2, 1))
        # latent linear transformation
        h_l = torch.permute(vq_latent, (0, 2, 1))
        h_l = self.conv(h_l) 
        h_l = torch.permute(h_l, (0, 2, 1))
        # transformer decoder 
        h = self.transformer_decoder(h_n, h_l)
        h = torch.permute(h, (0, 2, 1))
        h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_os = self.conv_o(h)
        re_os = torch.permute(re_os, (0, 2, 1))
        return {'re_observation_set': re_os}

## Vector quantizer
class TranVectorQuantizer(nn.Module):
    def __init__(self, args):
        # super(TranVectorQuantizer, self).__init__()
        super().__init__()
        self.args = args
        self.latent_size = tvq_latent_size
        self.embedding_size = tvq_embedding_size
        self.embedding_dim = tvq_embedding_dim
        self.codebook = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.embedding_size, 1/self.embedding_size)

    def forward(self, latent):
        # b_size = latent.size(0)
        latent_flattened = latent.contiguous().view(-1, self.embedding_dim) # [batch_size * forward_steps * player_num * latent_dim, embedding_dim]
        distances = (torch.sum(latent_flattened**2, dim=-1, keepdim=True)
                     + torch.sum(self.codebook.weight**2, dim=-1)
                     - 2 * torch.matmul(latent_flattened, self.codebook.weight.t()))

        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.embedding_size, device=latent.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_latent = torch.matmul(encodings, self.codebook.weight).view(latent.size())
        policy_vq_latent = latent + (quantized_latent - latent).detach()
        codebook_weight = self.codebook.weight.unsqueeze(0).expand(latent.size(0), -1, -1)
        return {'policy_vq_latent_set': policy_vq_latent, 'quantized_policy_latent_set': quantized_latent, 'codebook_set': codebook_weight}

## Action-Observation VQ-VAE
class AOTranVQVAE(nn.Module):
    def __init__(self, args, encoder, a_decoder, o_decoder, quantizer):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.quantizer = quantizer
        self.a_decoder = a_decoder
        self.o_decoder = o_decoder

    def forward(self, os_in, as_in, hidden=None):
        h_l = self.encoder(os_in, as_in)

        # Latent quantization
        re_q = self.quantizer(h_l['policy_latent_set'])
        vq_latent = re_q['policy_vq_latent_set'].unsqueeze(1).expand(-1, os_in.size(1), -1)
        memory = re_q['policy_vq_latent_set'].view(-1, tvq_latent_size, tvq_embedding_dim)

        noize_one_hot = F.one_hot(torch.randint(low=0, high=tvq_embedding_size, size=(os_in.size(0), tvq_noise_num))).to(torch.float)
        # Noise generation and Reparametrization Trick (detach)
        noise = torch.bmm(noize_one_hot, re_q['codebook_set']).detach() # codebook から一様乱数で取得した乱数

        re_p = self.a_decoder(vq_latent, os_in)
        re_s = self.o_decoder(noise, memory)
        return {**re_p, **re_s, **h_l, **re_q}

# RL with VQ-VAE wrapper model (ASC)
class TranVQASCModel(nn.Module):
    def __init__(self, args, hyperplane_n, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = TranVQEncoder(args, hyperplane_n)
        self.quantizer = TranVectorQuantizer(args)
        self.action_decoder = ActionConvVQDecoder(args, hyperplane_n)
        self.observation_decoder = ObservationTranVQDecoder(args, hyperplane_n)
        self.vae = AOTranVQVAE(args, self.encoder, self.action_decoder, self.observation_decoder, self.quantizer)

    def forward(self, x, hidden=None):
        os_in = x.get('os', None)
        as_in = x.get('as', None)
        latent = x.get('latent', None)
        # RL model
        out = self.rl_net(x)
        # State-Action VAE
        out_g = self.vae(os_in, as_in) if as_in!=None else None
        if out_g is not None:
            out = {**out, **out_g}
        # VAE action decoder
        out_g_p = self.action_decoder(os_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        return out



# base class of Environment

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.param = args['param'] #env_argsにparamを置いた場合
        self.depth = self.param['depth'] #深度(変更可能)
        self.hyperplane_n = self.param['hyperplane_n'] #超平面次元数(変更可能)
        self.treasure = np.array(self.param['treasure']) #報酬の場所(変更可能) #0番目は必ず(深度-1)になるように
        self.set_reward = self.param['set_reward'] #報酬の値(変更可能)
        self.other_reward = self.param['other_reward'] #treasure以外に到達した時の報酬設定
        self.start_random = self.param['start_random'] #初期地点をランダムにするか固定にするか / True: ランダムにする, False: 固定する
        self.pom_bool = self.param['pomdp_setting']['pom_bool'] #POMDPを導入するか / True: 導入する, False: 導入しない
        self.pom_state = self.param['pomdp_setting']['pom_state'] #途中報酬の座標
        self.pom_flag = 0 #途中報酬の座標を通ったら1, 通らなかったら0のまま．0だと報酬が得られない
        self.tree_s = []
        self.action_list = []
        self.goal_depth_all = [] #最大深度の状態を全て抽出
        self.goal_depth_place = [] #最大深度の状態をスタートがランダムでも到達できる座標のみ抽出
        self.place_list = []
        self.tree_make() #state_randomは使える
        self.place_list_make()#報酬が置ける位置の作成
        self.random_trasures_bool = self.param['random_trasures_setting']['random_trasures_bool'] #報酬の場所をランダムに設定するか / True: 設定する, False: 設定しない
        self.random_trasures_num = self.param['random_trasures_setting']['random_trasures_num'] #報酬の場所の個数
        self.random_reward_bool = self.param['random_reward_setting']['random_reward_bool']
        self.random_reward = self.param['random_reward_setting']['random_reward']
        self.random_reward_p = self.param['random_reward_setting']['random_reward_p']
        if self.start_random: #初期位置の確認
            self.treasure_error() #報酬が正しい場所に置けてるかの確認
        if self.random_trasures_bool:
            self.random_trasures()
        self.uns_bool = self.param['uns_setting']['uns_bool'] #非定常の有無
        self.uns_num = self.param['uns_setting']['uns_num'] #非定常の周期
        self.uns_count = 0
        if self.uns_bool:
            self.uns_make()
        # [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        # print(self.goal_depth_place)
        self.observation_noise = self.param['observation_noise']
        self.true_state = [] # 真の状態（ランダムな状態量を導入するとoutcomeの条件式がおかしくなる応急処置）
        self.jyotai_boolkari = self.param['jyotai_boolkari'] #ランダムな状態量を使用する場合の仮の変数
        if self.jyotai_boolkari | self.observation_noise > 0:
            self.state_qnp=[] # ランダムな状態量のnumpy配列
            self.state_quantity() # ランダムな状態量を生成
            self.tree_list = self.tree_np.tolist() # 通常の状態をリスト型にしたもの（理由は上記と同様, ランダムな状態量との対応付けに使用）
            #print(self.tree_np) # デバッグ用
            #print(self.state_qnp) # デバッグ用

    def Transition(self, action, state):
        """行動、状態を引数に次状態を返す"""
        if self.jyotai_boolkari: # ランダムな状態量を通常の状態に変換
            state = self.tree_np[self.state_qlist.index(state.tolist())]
        action = self.action_list_np[action]
        new_state_tmp = [state[i+1]+ action[0][i] for i in range(self.hyperplane_n)]
        new_state = (state[0]+1,) + tuple(new_state_tmp)
        new_state = np.array(new_state)
        self.true_state = new_state
        if self.jyotai_boolkari: # 通常の状態量をランダムな状態に変換
            new_state = self.state_qnp[self.tree_list.index(new_state.tolist())]
        return  new_state #新しい状態を返す

    def tree_make(self):
        for i in range(2,2+self.depth):
            coord_seed = list(range(0,i))
            coord = list(itertools.product(coord_seed, repeat = self.hyperplane_n))
            coordinate = [(i-2,) + t for t in coord]
            self.tree_s += coordinate
            if i == 2:
                self.action_list = coord #行動のリスト
            if i == 1+self.depth: #最大深度の状態を抽出
                self.goal_depth_all += coordinate
        if not self.start_random: #初期位置固定
            start = (-1,) + (0,) * self.hyperplane_n
            self.tree_s = np.insert(self.tree_s, 0, start, axis=0)
        self.tree_np = np.array(self.tree_s)
        self.action_list_np = np.array(self.action_list)

    def state_quantity(self):
        dt_now = datetime.datetime.now() # 時刻取得
        np.random.seed(dt_now.minute) # シード固定
        if self.jyotai_boolkari:
            self.state_qnp = np.random.randn(len(self.tree_np),self.hyperplane_n+1) # ランダムな状態量を生成
        if self.observation_noise == 1: # ノイズ小
            self.state_qnp = np.random.normal(0, 1, (len(self.tree_np),self.hyperplane_n+1))
            self.state_qnp += self.tree_np
        if self.observation_noise == 2: # ノイズ大
            self.state_qnp = np.random.normal(0, 5, (len(self.tree_np),self.hyperplane_n+1))
            self.state_qnp += self.tree_np
        #print(self.state_qnp) # デバッグ用
        self.state_qlist = self.state_qnp.tolist() # ランダムな状態量の配列をlist型に変換したもの（状態のインデックス取得するにはリスト型の方が都合がいい）

    def place_list_make(self):
        count = 0
        for i in range(self.depth):
            for j in range((2 + i)**(self.hyperplane_n)):
                if (((((2 + i)**(self.hyperplane_n)-i**self.hyperplane_n)/2)) <= j <  ((((2 + i)**(self.hyperplane_n)-i**self.hyperplane_n)/2) + i**self.hyperplane_n )) & (i != 0):
                    self.place_list.append(self.tree_np[count]) #報酬が置ける位置の配列の作成
                    if i == self.depth - 1:
                        self.goal_depth_place.append(self.tree_s[count])
                count = count + 1

    def treasure_error(self):
        for _ in range(len(self.treasure)):
            if not (self.treasure[_]==self.place_list).all(axis=1).any():#報酬が置けるかの判定
                print("Error: Treasure cannot be placed in that location.")
                print("Treasure is ",self.treasure[_],".")
                print("place_list",list(map(list,self.place_list ))) #報酬がおける位置の確認
                sys.exit("SystemExit: Treasure Error")

    def random_trasures(self):
        dt_now = datetime.datetime.now()
        random.seed(dt_now.minute)
        if self.start_random:
            self.treasure = np.array(random.sample(self.goal_depth_place,self.random_trasures_num))
        else:
            self.treasure = np.array(random.sample(self.goal_depth_all,self.random_trasures_num))
        print("new treasure random generation: ", self.treasure)

    def reset(self, args={}): #タスクリセット
        if self.start_random: #初期位置をランダムにする場合 (True)
            start = np.random.randint(len(self.action_list_np)) #初期座標ランダム選択    #start: 2*超平面次元通りある
            self.state = self.tree_np[start] #初期座標にリセット  #[0, x, y], start = 4;
            self.true_state = self.state
            if self.jyotai_boolkari:
                self.state = self.state_qnp[self.tree_list.index(self.state.tolist())]
            #print(self.state)
        else: #初期位置を固定にする場合 (False)
            self.state = self.tree_np[0] #[-1, 0], [-1, 0, 0]を最初に入れる
        if self.observation_noise > 0: # 観測ノイズの生成
            self.state_quantity()

    def play(self, action, player):
        #print("state :", self.state) # デバッグ用（現状態）
        #print("true_state :", self.true_state) # デバッグ用（真の現状態）
        a = np.array([action], )
        #print("action :", a) # デバッグ用（行動）
        next_s = self.Transition(a, self.state)
        self.state = next_s
        #print("next_state :", self.state) # デバッグ用（次状態）
        #print("next_true_state :", self.true_state) # デバッグ用（真の次状態）
        if self.pom_bool and np.array_equal(self.state, self.pom_state):
            self.pom_flag = 1

    def terminal(self):
        return self.true_state[0] == self.depth-1

    def uns_make(self):
        dt_now = datetime.datetime.now()
        random.seed(dt_now.minute)
        self.uns_trasures_list = np.array(random.choices(self.goal_depth_place, k = 10))
        print("uns_list : ", self.uns_trasures_list)

    def uns(self):
        print("tresure_before : ",self.treasure)
        self.treasure = np.array([self.uns_trasures_list[self.uns_count]])
        self.uns_count += 1
        print("new_tresure : ",self.treasure)

    def outcome(self):
        # 終端状態に到達した時に報酬を与える関数
        outcomes = [self.other_reward]
        #print(self.n)
        #self.n = self.n + 1
        #if self.n % 20000 == 0:
            #print("!!!n=",self.n)
        if self.pom_bool: #True
            if self.pom_flag and (self.state == self.treasure).all(axis=1).any(): #途中報酬の座標を通る && treasureの中にあるか
                outcomes = [self.set_reward]
            self.pom_flag = 0
        else: #False
            if self.random_reward_bool: #確率的な報酬
                if (self.state == self.treasure).all(axis=1).any():
                    treasure_num = np.where((self.treasure == self.state).all(axis=1))[0][0]
                    reward_choice = np.random.choice(self.random_reward[treasure_num], p=self.random_reward_p[treasure_num])
                    outcomes = [reward_choice]
            else:
                if (self.true_state == self.treasure).all(axis=1).any(): #treasureが2次元配列じゃないと動かない #(a==b)で同じshapeか，all(axis=1)で列方向に一致しているか，any()でどれか一つにでも当てはまるか，True・Falseを返す
                    treasure_num = np.where((self.treasure == self.true_state).all(axis=1))[0][0]
                    outcomes = [self.set_reward[treasure_num]]
        #if self.uns_bool:
            #if self.n % self.uns_num == 0:
                #print("n=",self.n)
                #print("tresure_before = ",self.treasure)
                #dt_now = datetime.datetime.now()
                #random.seed(dt_now.minute)
                #self.treasure = np.array(random.sample(self.goal_depth_place,self.random_trasures_num))
                #print("new_tresure= ",self.treasure)
        #print("now_goal:",self.state)
        #print("goal:",self.treasure)
        #print("return:",outcomes)
        #print("n =",self.n)
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    def net(self, args):
        agent_type = args['type']

        if agent_type == 'BASE':
            rl_model = SimpleModel(args, self.hyperplane_n)
        elif agent_type == 'QL' or agent_type == 'SAC':
            rl_model = PVQModel(args, self.hyperplane_n)
        elif agent_type == 'RSRS':
            rl_model = PVQCModel(args, self.hyperplane_n)
        else:
            rl_model = SimpleModel(args, self.hyperplane_n)
        # RND を任意の RL model に付随させる
        if args.get('use_RND', False):
            rl_model = RLwithRNDModel(args, self.hyperplane_n, rl_model)
        # ASC model
        asc_type = args.get('ASC_type', False)
        if asc_type == 'VAE':
            rl_model = ASCModel(args, self.hyperplane_n, rl_model)
        elif asc_type == 'VQ-VAE':
            rl_model = ASCVQModel(args, self.hyperplane_n, rl_model)
        elif asc_type == 'SeTranVAE':
            rl_model = TranASCModel(args, self.hyperplane_n, rl_model)
        elif asc_type == 'SeTranVQ-VAE':
            rl_model = TranVQASCModel(args, self.hyperplane_n, rl_model)
        
        return rl_model


    def observation(self, player=None):
        return self.state.astype(np.float32)

    def legal_actions(self, player):
        legal_actions = np.arange(2**self.hyperplane_n)
        return legal_actions

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            e.play()
        print(e)
        print(e.outcome())