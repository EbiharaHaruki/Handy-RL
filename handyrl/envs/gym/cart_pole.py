# OpenAI Gym

import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from collections import deque
from PIL import Image

from ...environment import BaseEnvironment

def get_screen(env):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(size=(84, 84)),
                    T.Grayscale(num_output_channels=1)])
    screen = resize(env.render())
    screen = np.expand_dims(np.asarray(screen), axis=2).transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen

class ConvNet(nn.Module):
    # Qlearning-conv
    def __init__(self, args, input_dim, action_num):
        super().__init__()
        C, H, W = 1, 84, 84
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]
        nn_size = 512
        neighbor_frames = 4

        self.conv1 = nn.Conv2d(in_channels=C*neighbor_frames, out_channels=16, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_sizes[1], stride=self.strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_sizes[2], stride=self.strides[2])
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(W, n_layer=3)
        convh = self.conv2d_size_out(H, n_layer=3)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, nn_size)
        self.head_p = nn.Linear(nn_size, action_num)
        self.head_v = nn.Linear(nn_size, 1)
        self.head_a = nn.Linear(nn_size, action_num) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h = F.relu(self.bn1(self.conv1(o)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h_l = self.fc1(h.view(h.size(0), -1))
        h = F.relu(h_l)
        p = self.head_p(h)
        v = self.head_v(h)
        a = self.head_a(h)
        b = self.head_b(h)
        q = b + a - a.sum(-1).unsqueeze(-1)

        return {
            'policy': p, 'value': v, 
            'advantage_for_q': a, 'qvalue': q, 'rl_latent': h_l}

    def conv2d_size_out(self, size, n_layer):
        cnt = 0
        size_out = size
        while cnt < n_layer:
            size_out = (size_out - self.kernel_sizes[cnt]) // self.strides[cnt] + 1
            cnt += 1
        return size_out

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
        nn_size = 128
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.fc2 = nn.Linear(nn_size, nn_size)
        self.head_p = nn.Linear(nn_size, action_num) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, action_num) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h_l = self.fc1(o)
        h = F.relu(h_l)
        h_l = self.fc2(h)
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
        nn_size = 128
        #self.fc1 = nn.Linear(input_dim, nn_size)
        self.fc1 = nn.Linear(input_dim, nn_size)
        self.fc2 = nn.Linear(nn_size, nn_size)
        self.head_p = nn.Linear(nn_size, action_num) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, action_num) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        self.head_c = nn.Linear(nn_size, action_num) # 信頼度(confidence rate)

    def forward(self, x, hidden=None):
        o = x.get('o', None)
        h_l = self.fc1(o)
        h_l = F.relu(h_l)
        h_l = self.fc2(h_l)
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

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.env = gym.make('CartPole-v0', render_mode="rgb_array")
        self.neighbor_frames = 1
        self.frames = None
        self.latest_obs = None
        self.total_reward = 0
        self.done, self.truncated = False, False
        # 以下は画像の時
        """super().__init__()
        self.env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped #'CartPole-v1'
        self.neighbor_frames = 4
        self.frames = None
        self.latest_obs = None
        self.total_reward = 0
        self.done, self.truncated = False, False"""
        # 入力特徴次元数
        self.input_dim = 4
        # action 数
        self.action_num = 2

    def reset(self, args={}):
        self.latest_obs,_ = self.env.reset()
        self.total_reward = 0
        self.done, self.truncated = False, False
        # 以下は画像の時
        """self.env.reset()
        frame = get_screen(self.env)
        self.frames = deque([frame]*self.neighbor_frames, maxlen=self.neighbor_frames)
        self.latest_obs = np.stack(self.frames, axis=1)[0,:]
        self.total_reward = 0
        self.done, self.truncated = False, False"""

    def play(self, action, player):
        observation, reward, done, truncated, info = self.env.step(action)
        self.latest_obs = observation
        self.latest_reward = reward
        self.done = done
        self.truncated = truncated
        self.latest_info = info
        self.total_reward += reward
        # 以下は画像の時
        """observation, reward, done, truncated, info = self.env.step(action)
        frame = get_screen(self.env)
        self.frames = deque([frame]*self.neighbor_frames, maxlen=self.neighbor_frames)
        self.latest_obs = np.stack(self.frames, axis=1)[0,:]
        self.latest_reward = reward
        self.done = done
        self.truncated = truncated
        self.latest_info = info
        self.total_reward += reward"""

    def terminal(self):
        return self.done or self.truncated

    def outcome(self):
        outcomes = [self.total_reward]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    # Neural network 構築
    def net(self, args):
        agent_type = args['subtype'] if args['type'] == 'ASC' else args['type']

        if agent_type == 'BASE':
            rl_model = SimpleModel(args, self.input_dim, self.action_num)
        elif agent_type == 'QL' or agent_type == 'SAC':
            rl_model = PVQModel(args, self.input_dim, self.action_num)
            #rl_model = ConvNet(args, self.input_dim, self.action_num)
        elif agent_type == 'RSRS':
            rl_model = PVQCModel(args, self.input_dim, self.action_num)
        elif agent_type == 'R4D-RSRS':
            rl_model = R4DPVQCModel(args, self.input_dim, self.action_num)
        elif agent_type == 'EMA-RSRS':
            rl_model = PVQCModel(args, self.input_dim, self.action_num)
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

    def observation(self, player=None):
        return self.latest_obs

    def action_length(self):
        return self.env.action_space.n

    def legal_actions(self, player=None):
        return np.arange(self.action_length())


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        print(_)
        e.reset()
        while not e.terminal():
            e.env.render()
            actions = e.legal_actions()
            e.play(random.choice(actions))
    e.env.close()