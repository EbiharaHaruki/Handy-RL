# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of simple pyramid

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
import math
import os

from torch.nn import init
import torch.optim as optim

from ..environment import BaseEnvironment


class SimpleModel(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.head_p = nn.Linear(nn_size, action_num)
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
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.head_p = nn.Linear(nn_size, action_num) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, action_num) # advantage
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
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.head_p = nn.Linear(nn_size, action_num) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, action_num) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        self.head_c = nn.Linear(nn_size, action_num) # 信頼度(confidence rate)

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


# RSRS with R4D
class R4DPVQCModel(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        confidence_size = 32
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.head_p = nn.Linear(nn_size, action_num) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, action_num) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        # 信頼度
        ## 学習
        self.fc_c = nn.Linear(input_dim, nn_size)
        self.head_c = nn.Linear(nn_size, action_num * confidence_size)
        ## 固定
        self.fc_c_fix = nn.Linear(input_dim, nn_size)
        self.head_c_fix = nn.Linear(nn_size, action_num * confidence_size)

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h_l = self.fc1(o)
        h = F.relu(h_l)
        p = self.head_p(h)
        v = self.head_v(h)
        a = self.head_a(h)
        b = self.head_b(h)
        q = b + a - a.sum(-1).unsqueeze(-1)
        h_c = F.relu(self.fc_c(o))
        c = self.head_c(h_c)
        h_c_fix = F.relu(self.fc_c_fix(o))
        c_fix = self.head_c_fix(h_c_fix)
        return {
            'policy': p, 'value': v, 
            'advantage_for_q': a, 'qvalue': q, 'rl_latent': h_l, 
            'confidence_57': c, 'confidence_57_fix': c_fix}


# RND
class RNDModel(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        # RND head dim : 探索の複雑性
        rnd_head_size = 16
        ## 学習
        self.fc_rnd = nn.Linear(input_dim, nn_size)
        self.head_rnd = nn.Linear(nn_size, rnd_head_size)
        ## 固定
        self.fc_rnd_fix = nn.Linear(input_dim, nn_size)
        self.head_rnd_fix = nn.Linear(nn_size, rnd_head_size)

    def forward(self, o_in, hidden=None):
        rnd = F.relu(self.fc_rnd(o_in))
        rnd = self.head_rnd(rnd)
        rnd_fix = F.relu(self.fc_rnd_fix(o_in))
        rnd_fix = self.head_rnd_fix(rnd_fix)
        return {'embed_state': rnd, 'embed_state_fix': rnd_fix}


# RL with RND wrapper model
class RLwithRNDModel(nn.Module):
    def __init__(self, args, input_dim, action_num, rl_net):
        super().__init__()
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.rnd_net = RNDModel(args, input_dim, action_num)

    def forward(self, x, hidden=None):
        o_in = x.get('o', None)
        # RL model
        out = self.rl_net(x)
        # rnd
        out_rnd = self.rnd_net(o_in)
        out = {**out, **out_rnd}
        return out


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim, dropout = 0.1, max_len = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-math.log(500.0) / latent_dim))
        positional_encoding = torch.zeros(1, max_len, latent_dim)
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)



# VAE + Transformer (SeTranVAE) パラメータ 
tvae_latent_dim = 32 # 潜在変数の次元数
tvae_emb_size = 64 # transformer に入力する潜在変数を線形変換したサイズ
tvae_ffn_size = 256 # transformer に FFN 中間ユニット数
tvae_cnn_size = [128, 256, 128, 64] # action decoder の中間ユニット数 
tvae_head_num = 2 # ヘッド数
tvae_tf_layer_num = 3 # transformer のレイヤー数
tvae_noise_num = 9 # observation decoder の target 入力するノイズ数

# VAE + Transformer
## Encoder
class TranEncoder(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        self.action_num = action_num
        self.feature_num = input_dim + self.action_num
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
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=(tvae_latent_dim + input_dim), out_channels=tvae_cnn_size[0], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=tvae_cnn_size[0], out_channels=tvae_cnn_size[1], kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=tvae_cnn_size[1], out_channels=tvae_cnn_size[2], kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=tvae_cnn_size[2], out_channels=tvae_cnn_size[3], kernel_size=1)
        self.conv_p = nn.Conv1d(in_channels=tvae_cnn_size[3], out_channels=action_num, kernel_size=1)
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
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(tvae_latent_dim), out_channels=tvae_emb_size, kernel_size=1)
        self.conv_o= nn.Conv1d(in_channels=tvae_emb_size, out_channels=input_dim, kernel_size=1)
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
    def __init__(self, args, input_dim, action_num, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = TranEncoder(args, input_dim, action_num)
        self.action_decoder = ActionConvDecoder(args, input_dim, action_num)
        self.observation_decoder = ObservationTranDecoder(args, input_dim, action_num)
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
            out['ASC'] = out_g
        # VAE action decoder
        out_g_p = self.action_decoder(os_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        return out


# VQ-VAE + Transformer (VQ-SeTranVAE) パラメータ
tvq_latent_size= 8 # lattent の個数, lattent の次元数そのものは vq_latent_dim * vq_embedding_dim
tvq_codebook_size = 128 # codebook の code (embedding vector) のエントリ数
tvq_embedding_dim = 32 # embedding vector 1 つずつの長さ

tvq_emb_size = 64 # transformer に入力する潜在変数を線形変換したサイズ
tvq_ffn_size = 256 # transformer に FFN 中間ユニット数
tvq_cnn_size = [128, 256, 128, 64] # action decoder の中間ユニット数 
tvq_head_num = 2 # ヘッド数
tvq_tf_layer_num = 2 # transformer のレイヤー数
tvq_noise_num = 8 # observation decoder の target 入力するノイズ数
## EMA parameter
tvq_decay = 0.99
tvq_epsilon = 1e-5

# VQ-VAE + Transformer
## Encoder
class VQTranEncoder(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        self.action_num = action_num
        self.feature_num = input_dim + self.action_num
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
class ActionVQConvDecoder(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv1 = nn.Conv1d(in_channels=(tvq_latent_size * tvq_embedding_dim + input_dim), out_channels=tvq_cnn_size[0], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=tvq_cnn_size[0], out_channels=tvq_cnn_size[1], kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=tvq_cnn_size[1], out_channels=tvq_cnn_size[2], kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=tvq_cnn_size[2], out_channels=tvq_cnn_size[3], kernel_size=1)
        self.conv_p = nn.Conv1d(in_channels=tvq_cnn_size[3], out_channels=action_num, kernel_size=1)
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
        h = self.relu(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.dropout(h)
        re_ps = self.conv_p(h)
        re_ps = torch.permute(re_ps, (0, 2, 1))
        return {'re_policy_set': re_ps}

## Observation decoder
class ObservationVQTranDecoder(nn.Module):
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        self.args = args
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels=(tvq_embedding_dim), out_channels=tvq_emb_size, kernel_size=1)
        self.conv_o= nn.Conv1d(in_channels=tvq_emb_size, out_channels=input_dim, kernel_size=1)
        # # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, tvq_latent_size, tvq_emb_size))
        # Transformer Encoder
        # decoder_layer = nn.TransformerDecoderLayer(tvq_emb_size, tvq_head_num, tvq_ffn_size, dropout=self.args['ASC_dropout'], batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(tvq_emb_size, tvq_head_num, tvq_ffn_size, dropout=0.0, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, tvq_tf_layer_num)
        self.bn = nn.BatchNorm1d(tvq_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args['ASC_dropout'])
        self.p_encoding = PositionalEncoding(tvq_emb_size, self.args['ASC_dropout'], tvq_latent_size)

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
        h = self.transformer_decoder(h_n, self.p_encoding(h_l))
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
        self.codebook_size = tvq_codebook_size
        self.embedding_dim = tvq_embedding_dim
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)

    def forward(self, latent, generating_from_uniform=False):
        if generating_from_uniform:
            # ランダムにコードブックから生成する
            noize_one_hot = F.one_hot(torch.randint(low=0, high=tvq_codebook_size, size=(1, tvq_latent_size))).to(torch.float)
            quantized_latent = torch.bmm(noize_one_hot, self.codebook.weight).view(1, 1, -1).detach()

            return {'policy_vq_latent_set': None, 'quantized_policy_latent_set': quantized_latent, 'codebook_set': self.codebook.weight}    
        else:
            # b_size = latent.size(0)
            latent_flattened = latent.contiguous().view(-1, self.embedding_dim) # [batch_size * forward_steps * player_num * latent_dim, codebook_dim]
            distances = (torch.sum(latent_flattened**2, dim=-1, keepdim=True)
                        + torch.sum(self.codebook.weight**2, dim=-1)
                        - 2 * torch.matmul(latent_flattened, self.codebook.weight.t()))

            encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.size(0), self.codebook_size, device=latent.device)
            encodings.scatter_(1, encoding_indices, 1)
            # 量子化された潜在ベクトル
            quantized_latent = torch.matmul(encodings, self.codebook.weight).view(latent.size())
            # 勾配を回避するための量子化された潜在ベクトル（値は quantized_latent と同じ）
            policy_vq_latent = latent + (quantized_latent - latent).detach()
            # Transformer decoder の乱数に対応づけるため codebook も batch サイズに拡張して出力する
            codebook_weight = self.codebook.weight.unsqueeze(0).expand(latent.size(0), -1, -1)
            return {'policy_vq_latent_set': policy_vq_latent, 'quantized_policy_latent_set': quantized_latent, 'codebook_set': codebook_weight}

## EMA Vector quantizer
class EMATranVectorQuantizer(nn.Module):
    def __init__(self, args):
        # super(TranVectorQuantizer, self).__init__()
        super().__init__()
        self.args = args
        self.latent_size = tvq_latent_size
        self.codebook_size = tvq_codebook_size
        self.embedding_dim = tvq_embedding_dim
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)

        self.decay = tvq_decay
        self.epsilon = tvq_epsilon

        self.register_buffer('ema_cluster_size', torch.zeros(self.codebook_size))
        self.ema_w = nn.Parameter(torch.Tensor(self.codebook_size, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self, latent, generating_from_uniform=False):
        if generating_from_uniform:
            # ランダムにコードブックから生成する
            noize_one_hot = F.one_hot(torch.randint(low=0, high=self.codebook_size, size=(1, self.latent_size)), num_classes=self.codebook_size).to(torch.float)
            quantized_latent = torch.bmm(noize_one_hot, self.codebook.weight.unsqueeze(0)).view(1, 1, -1).detach()

            return {'policy_vq_latent_set': None, 'quantized_policy_latent_set': quantized_latent, 'codebook_set': self.codebook.weight}    
        else:
            # b_size = latent.size(0)
            latent_flattened = latent.contiguous().view(-1, self.embedding_dim) # [batch_size * forward_steps * player_num * latent_dim, codebook_dim]
            distances = (torch.sum(latent_flattened**2, dim=-1, keepdim=True)
                        + torch.sum(self.codebook.weight**2, dim=-1)
                        - 2 * torch.matmul(latent_flattened, self.codebook.weight.t()))

            encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.size(0), self.codebook_size, device=latent.device)
            encodings.scatter_(1, encoding_indices, 1)
            # 量子化された潜在ベクトル
            quantized_latent = torch.matmul(encodings, self.codebook.weight).view(latent.size())

            # EMA codebook update
            if self.training:
                # batch 内の codebook の各要素の選択回数
                encodings_sum = encodings.sum(0)
                # batch 内の lattent をマッピング
                distance_w = torch.matmul(encodings.t(), latent_flattened)
                # codebook の各要素の選択回数を更新（過去の batch 毎の選択回数の EMA）
                self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings_sum
                # codebook の各要素の Target となる weight を更新
                self.ema_w = nn.Parameter(self.decay * self.ema_w + (1 - self.decay) * distance_w)
                # 正規化と安定化
                n = torch.sum(self.ema_cluster_size)
                cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon)) * n
                self.codebook.weight.data = self.ema_w / cluster_size.unsqueeze(1)

            # 勾配を回避するための量子化された潜在ベクトル（値は quantized_latent と同じ）
            policy_vq_latent = latent + (quantized_latent - latent).detach()
            # Transformer decoder の乱数に対応づけるため codebook も batch サイズに拡張して出力する
            codebook_weight = self.codebook.weight.unsqueeze(0).expand(latent.size(0), -1, -1)
            return {'policy_vq_latent_set': policy_vq_latent, 'quantized_policy_latent_set': quantized_latent, 'codebook_set': codebook_weight}

## Action-Observation VQ-VAE
class AOVQTranVAE(nn.Module):
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

        noize_one_hot = F.one_hot(torch.randint(low=0, high=tvq_codebook_size, size=(os_in.size(0), tvq_noise_num)), num_classes=tvq_codebook_size).to(torch.float)
        # Noise generation and Reparametrization Trick (detach)
        noise = torch.bmm(noize_one_hot, re_q['codebook_set']).detach() # codebook から一様乱数で取得した乱数

        re_p = self.a_decoder(vq_latent, os_in)
        re_s = self.o_decoder(noise, memory)
        return {**re_p, **re_s, **h_l, **re_q}

# RL with VQ-VAE wrapper model (ASC)
class VQTranASCModel(nn.Module):
    def __init__(self, args, input_dim, action_num, rl_net):
        super().__init__()
        self.args = args
        # RL モデル関連
        self.rl_net = rl_net
        ## 生成モデル関係
        self.encoder = VQTranEncoder(args, input_dim, action_num)
        self.quantizer = EMATranVectorQuantizer(args)
        self.action_decoder = ActionVQConvDecoder(args, input_dim, action_num)
        self.observation_decoder = ObservationVQTranDecoder(args, input_dim, action_num)
        self.vae = AOVQTranVAE(args, self.encoder, self.action_decoder, self.observation_decoder, self.quantizer)

    def forward(self, x, hidden=None):
        os_in = x.get('os', None)
        as_in = x.get('as', None)
        latent = x.get('latent', None)
        generating = x.get('generating', False)
        # RL model
        out = self.rl_net(x)
        # State-Action VAE
        out_g = self.vae(os_in, as_in) if as_in!=None else None
        if out_g is not None:
            out['ASC'] = out_g
        # VAE action decoder
        out_g_p = self.action_decoder(os_in, latent) if latent!=None else None
        if out_g_p is not None:
            out = {**out, **out_g_p}
        if generating:
            o_in = (x.get('o', None)).unsqueeze(1)
            vq_l = self.quantizer(None, True)
            re_policy = self.action_decoder(vq_l['quantized_policy_latent_set'], o_in)
            out = {**out, **re_policy}
        return out


# Pyramid nodes
class Node():
    def __init__(self, depth=0, hyperplane_coordinates=np.array([0]), center=None, features=None, obs_var=0.0, rng=np.random.default_rng(), reward_func=(lambda a, b, c, d: 0), pomdp=False):
        self.depth = depth
        self.coordinates = np.append(hyperplane_coordinates, depth)
        # 状態特徴量の設定
        self.features = self.coordinates if features is None else center + features
        self.features_dim = self.features.size
        self.action_num = 2**hyperplane_coordinates.size
        self.obs_var = obs_var # 観測ノイズの分散
        # 報酬関係
        self.reward_func = reward_func
        self.r_type = None
        self.r_p = 0.0
        self.r_m = 0.0
        self.r_v = 0.0
        self.rng = rng
        # e.g. hyperplane_dim == 2 -> children is [[0,1], [2,3]]
        self.children = np.empty(self.action_num, dtype=object)
        self.is_terminal = False
        self.is_key = False
        self.is_pomdp = pomdp
        # 訪問カウント
        self.visit_count = 0


    # action index に対して次状態ノードを結合
    def connect_node(self, child, action_index):
        self.children[action_index] = child

    # 報酬関数の設置
    def set_reward_func(self, reward_func):
        self.reward_func = reward_func

    # 報酬パラメータの設置
    def set_reward_param(self, reward_param):
        self.r_type = reward_param.get('type', None)
        self.r_m = reward_param.get('mu', 0.0)
        self.r_v = reward_param.get('var', 0.0)

    # POMDP における報酬の鍵になる状態として設定
    def set_key(self):
        self.is_key = True        
    
    # 終端格納
    def set_terminal(self):
        self.is_terminal = True

    # ノイズを乗せた状態観測
    def observed(self):
        if self.obs_var == 0:
            return self.features
        else:
            noise = self.rng.normal(0, self.obs_var, self.features.size)
            return self.features + noise

    # 状態の訪問回数を更新    
    def count(self):
        self.visit_count += 1

    # 状態遷移
    def transit(self, action_index):
        return self.children[action_index]

    # 報酬観測
    def reward(self, lock_open = True):
        if lock_open or self.is_pomdp:
            return self.reward_func(self.rng, self.r_type, self.r_m, self.r_v)
        else:
            return 0


class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.param = args['param'] #env_argsにparamを置いた場合
        self.depth = self.param['depth'] #深度(変更可能)
        self.hyperplane_dim = self.param['hyperplane_dim'] #超平面次元数(変更可能)

        # 報酬やノイズなどに使用する乱数発生器
        self.general_seed = self.param['general_seed'] # 特徴量関係の乱数 seed（保存対象）
        if self.general_seed < 0: # seed が 0 未満なら時間乱数から取る
            dt_now = datetime.datetime.now()
            self.general_seed =  (dt_now.hour * 60 * 60) + (dt_now.minute * 60) + dt_now.second # 非保存対象の乱数 seed
        self.general_rng = np.random.default_rng(self.general_seed)

        # 特徴量設定
        self.feature_seed = self.param['features']['seed'] # 特徴量関係の乱数 seed（保存対象）
        if self.feature_seed < 0: # seed が 0 未満なら時間乱数から取る
            dt_now = datetime.datetime.now()
            self.feature_seed = dt_now.microsecond # 特徴量関係の乱数 seed（保存対象）
        self.feature_rng = np.random.default_rng(self.feature_seed) # 読み込み再現可能な状態特徴量初期化用の乱数発生器
        self.feature_dim = self.param['features']['dim'] # 0 だと特徴量が座標そのものになる, 1 以上なら任意の次元数の座標とは異なる特徴量が設定される
        self.feature_f = self.feature_func if self.feature_dim != 0 else None # 特徴量関係の乱数発生器
        self.obs_var = self.param['features']['obs_var'] # 状態特徴の観測時に乗るノイズ（標準正規分布の分散）

        # 報酬設定
        self.reward_params = []
        for i in range(len(self.param['rewards']['depth'])):
            reward_param = {
                'depth': self.param['rewards']['depth'][i],
                'coordinates': tuple(self.param['rewards']['coordinates'][i]),
                'type': self.param['rewards']['type'][i],
                'mu': self.param['rewards']['mu'][i],
                'var': self.param['rewards']['var'][i]
            }
            self.reward_params.append(reward_param)
        self.reward_params = self.set_reward_coordinates(self.hyperplane_dim, self.reward_params, self.feature_rng)

        # 報酬の鍵設定 (POMDP)
        self.is_pomdp = (len(self.param['keys']['depth']) != 0) # POMDP 環境下かどうか
        self.key_params = []
        for i in range(len(self.param['keys']['depth'])):
            key_param = {
                'depth': self.param['keys']['depth'][i],
                'coordinates': tuple(self.param['keys']['coordinates'][i]),
            }
            self.key_params.append(key_param)
        self.key_num = len(self.key_params) # 報酬の鍵の鍵の数

        # 非定常環境設定
        self.shift_type = self.param['shift']['type']
        self.is_shit_env = False if self.shift_type == 'none' else True
        self.shift_param = {
            'intercept': self.param['shift'].get('intercept', 0.0), # 環境変異の特徴量変換の際の切片
            'slope': self.param['shift'].get('slope', 1.0), # 環境変異の特徴量変換の際の傾き
            'interval_episodes': self.param['shift'].get('interval_episodes', -1) # 環境変異の間隔
        }
        self.shift_count = 0

        # node 生成
        self.nodes = self.create_tree(self.depth, self.hyperplane_dim, self.reward_params, \
                                      self.general_rng, self.feature_rng, \
                                      self.obs_var, self.feature_f, \
                                      self.is_pomdp, self.key_params)

        # 現在 node 初期化
        self.start_random = self.param['start_random'] #初期地点をランダムにするか固定にするか / True: ランダムにする, False: 固定する
        self.initial_node = self.nodes[0][self.feature_rng.integers(self.nodes[0].size)]
        self.current_node = self.initial_node
        self.current_node.count()

        # 入力特徴次元数
        self.input_dim = self.current_node.features_dim
        # action 数
        self.action_num = self.current_node.action_num

        # 報酬の鍵の鍵の入手数の初期化
        self.got_key_num = 0
        # 報酬入手可能かの初期化
        self.lock_open = False

    ## 状態特徴量生成器
    def feature_func(self):
        return self.feature_rng.uniform(-1, 1, self.feature_dim)

    # 報酬関数
    def reward_func(self, _rng, _r_type, _mu, _var):
        if _r_type == 'binominal':
            return _rng.binomial(1, _mu, 1)[0]
        elif _r_type == 'normal':
            return _rng.normal(_mu, _var, 1)[0]
        elif _r_type == 'fix':
            return _mu
        else:
            return 1

    # 報酬設置関数
    def set_reward_coordinates(self, hyperplane_dim, reward_params, feature_rng):
        for i, r in enumerate(reward_params):
            coor = r['coordinates']
            if  coor == (): # 報酬箇所が空 tuple () の場合はランダム設置する

                count = 0
                while count < 100: 
                    count += 1
                    c = tuple(feature_rng.integers(1, r['depth']-1, (hyperplane_dim,)))
                    is_original = True
                    # 報酬箇所が重複していない事(is_original = True)を確かめる
                    for j in range(i):
                        if c == reward_params[j]['coordinates']:
                            is_original = False #重複している
                            break;
                    if is_original:
                        reward_params[i]['coordinates'] = c
                        break;
        return reward_params

    # 超平面の結合関数
    def create_nd_grid(self, arrays):
        grids = np.meshgrid(*arrays)
        return np.stack(grids, axis=-1)

    # 格子空間の等分割中心座標を抽出する関数
    def grid_centers_specific(self, n, m):
        # 各次元を等しく分割するための最適なサイズを計算
        divisions_per_dim = int(np.ceil(m ** (1 / n)))
        
        # 全ての次元で等しい分割を行い、格子の中心座標を計算
        linspace = np.linspace(-1 + 1 / divisions_per_dim, 1 - 1 / divisions_per_dim, divisions_per_dim)
        
        # 各次元の格子の組み合わせを取得
        grids = np.meshgrid(*[linspace] * n)
        
        # 格子中心座標を整形
        grid_centers = np.vstack([np.ravel(grid) for grid in grids]).T
        return grid_centers

    # ノード構築
    def create_tree(self, depth, hyperplane_dim, reward_params, general_rng, feature_rng, obs_var=0.0, f=None, is_pomdp=True, key_params=[]):
        nodes = np.empty(depth, dtype=object)
        grid_nodes = np.empty(depth, dtype=object)
        
        # ランダム特徴量を使う場合は depth ごとの中心座標を決めていく (depth 情報を暗黙的に含むように)
        if f is not None:
            test_feature = f()
            # 格子空間を等分割した中心座標
            grid_centers = self.grid_centers_specific(test_feature.size, depth)
            # 中心座標から depth の数だけ非復元抽出
            centers = feature_rng.choice(grid_centers, size=depth, replace=False)*2
        
        for d in range(depth):
            _d = d+1
            flat_length = (_d+1)**hyperplane_dim
        
            # 等差数列をリストにまとめる
            arrays = [np.linspace(-_d/2, _d/2, _d+1) for _ in range(hyperplane_dim)]
            # hyperplane_dim 次元の格子空間座標を作成
            grid_coordinates = self.create_nd_grid(arrays)
            # flat 化
            flat_coordinates = grid_coordinates.reshape(flat_length, hyperplane_dim)
            
            # node 生成
            if f is None:
                # 状態特徴量は格子空間座標そのまま
                tmp_nodes = np.array([Node(depth=d, hyperplane_coordinates=coordinates, obs_var=obs_var, rng=general_rng, pomdp=is_pomdp) for coordinates in flat_coordinates])
            else:
                # ランダムかつ好きな長さの状態特徴量
                tmp_nodes = np.array([Node(depth=d, hyperplane_coordinates=coordinates, obs_var=obs_var, rng=general_rng, pomdp=is_pomdp, center=centers[d], features=f()) for i, coordinates in enumerate(flat_coordinates)])
            # flat に展開したノード
            nodes[d] = tmp_nodes
            # 超平面上に展開したノード
            grid_nodes[d] = tmp_nodes.reshape(grid_coordinates.shape[0:-1])
            # 報酬設置
            for r in reward_params:
                if (d+1) == r['depth']:
                    grid_nodes[d][r['coordinates']].set_reward_func(self.reward_func)
                    grid_nodes[d][r['coordinates']].set_reward_param(r)
            # 報酬の鍵を設置 (POMDP)
            for k in key_params:
                if (d+1) == k['depth']:
                    grid_nodes[d][k['coordinates']].set_key()
            # ノード接続
            if d != 0:
                for i, n in enumerate(nodes[d-1]):
                    i_indices = np.array(np.unravel_index(i, grid_nodes[d-1].shape))
                    for j in range(n.action_num):
                        j_indices = np.array(np.unravel_index(j, grid_nodes[0].shape))
                        indices = tuple(i_indices + j_indices) 
                        n.connect_node(grid_nodes[d][indices], j) 
                        if d == depth - 1:
                            grid_nodes[d][indices].set_terminal()
        return nodes

    # タスクリセット
    def reset(self, args={}): 
        if self.start_random: #初期位置をランダムにする場合 (True)
            self.initial_node = self.nodes[0][self.feature_rng.integers(self.nodes[0].size)]
        # 現在 node 初期化
        self.current_node = self.initial_node
        self.current_node.count()
        # 報酬の鍵の鍵の入手数の初期化
        self.got_key_num = 0
        # 報酬入手可能かの初期化
        self.lock_open = False

    # 行動の実行
    def play(self, action, player):
        self.current_node = self.current_node.transit(action)
        self.current_node.count()
        self.got_key_num += self.current_node.is_key

    # 終端状態チェック
    def terminal(self):
        return self.current_node.is_terminal

    # 終端報酬
    def outcome(self):
        outcomes = [self.current_node.reward()]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    # 状態観測
    def observation(self, player=None):
        return (self.current_node.observed()).astype(np.float32)

    # 合法手出力
    def legal_actions(self, player):
        return np.arange(self.action_num)

    # 現在 Prayer => 常に Player 0  
    def players(self):
        return [0]

    # 非定常環境における環境遷移
    def shift_env(self, num_episodes):
        self.shift_count += 1
        return False

    def fprint_env_status(self, role, worker_id=None):
        dirname = 'env_status_log'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if role == 't':
            role_str = 'train_'
        elif role == 'g':
            role_str = 'generation_'
        elif role == 'e':
            role_str = 'evaluation_'
        else:
            role_str = ''
        logname = 'simple_pyramid_' + role_str + ('' if worker_id is None else 'wid' + str(worker_id) + '_') + current_time + '.csv'
        filename = os.path.join(dirname, logname)

        with open(filename, 'a') as file:
            file.write(f'general_seed, feature_seed\n')
            file.write(f'{self.general_seed}, {self.feature_seed}\n')
            file.write(f'depth, index, coordinates, features, reward, visit_count\n')

        for d in range(self.depth):
            _d = d+1
            for i, node in enumerate(self.nodes[d]):
                # 全てのノードの訪問回数を depth と共に保存
                reward = '-' if node.r_type is None else node.r_m
                with open(filename, 'a') as file:
                    file.write(f'{_d}, {i}, {node.coordinates}, {node.features}, {reward}, {node.visit_count}\n')
        return True

    # seed の出力
    def get_seed(self):
        return {'general_seed': self.general_seed, 'feature_seed': self.feature_seed}

    # Neural network 構築
    def net(self, args):
        agent_type = args['subtype'] if args['type'] == 'ASC' else args['type']

        if agent_type == 'BASE':
            rl_model = SimpleModel(args, self.input_dim, self.action_num)
        elif agent_type == 'QL' or agent_type == 'SAC':
            rl_model = PVQModel(args, self.input_dim, self.action_num)
        elif agent_type == 'RSRS':
            rl_model = PVQCModel(args, self.input_dim, self.action_num)
        elif agent_type == 'R4D-RSRS':
            rl_model = R4DPVQCModel(args, self.input_dim, self.action_num)
        else:
            rl_model = SimpleModel(args, self.input_dim, self.action_num)
        # RND を任意の RL model に付随させる
        if args.get('use_RND', False):
            rl_model = RLwithRNDModel(args, self.input_dim, self.action_num, rl_model)
        # ASC model
        asc_type = args.get('ASC_type', '')
        if asc_type == 'SeTranVAE':
            rl_model = TranASCModel(args, self.input_dim, self.action_num, rl_model)
        elif asc_type == 'VQ-SeTranVAE':
            rl_model = VQTranASCModel(args, self.input_dim, self.action_num, rl_model)
        
        return rl_model

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            e.play()
        print(e)
        print(e.outcome())