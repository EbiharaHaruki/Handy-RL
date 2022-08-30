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


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(
            filters0, filters1, kernel_size,
            stride=1, padding=kernel_size//2, bias=bias
        )
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):
    def __init__(self, input_size, out_filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters

        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h


class SimpleModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 256
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n)
        self.head_v = nn.Linear(nn_size, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        return {'policy': h_p, 'value': torch.tanh(h_v)}


class CountBasedModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 2096
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n)
        self.head_v = nn.Linear(nn_size, 1)
        self.count_dict = {}

    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        coord_array = []
        for i in x[0]: coord_array.append(int(i.to('cpu').detach().numpy().copy()))
        coord_array = tuple(coord_array)
        if coord_array in self.count_dict: self.count_dict[coord_array] += 1
        else: self.count_dict[coord_array] = 1
        h_v += 1 / (2 * np.sqrt(self.count_dict[coord_array]))
        return {'policy': h_p, 'value': torch.tanh(h_v)}


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TargetModel(nn.Module):
    def __init__(self, hyperplane_n):
        super(TargetModel, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(hyperplane_n+1, 512)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        target_feature = self.target(obs)
        return target_feature


class PredictorModel(nn.Module):
    def __init__(self, hyperplane_n):
        super(PredictorModel, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hyperplane_n+1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        for param in self.predictor.parameters():
            param.requires_grad = True

    def forward(self, obs):
        obs.requires_grad = True
        predict_feature = self.predictor(obs)
        return predict_feature


class RNDModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 2048
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n)
        self.head_v = nn.Linear(nn_size, 1)
        self.hyperplane_n = hyperplane_n
        self.learning_rate = 1e-6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = TargetModel(hyperplane_n).to(self.device)
        self.predictor_model = PredictorModel(hyperplane_n).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.predictor_model.parameters()), lr=self.learning_rate)

    def get_intrinsic_rewards(self, x):
        self.predictor_model.eval()
        target_value = self.target_model(x)
        predictor_value = self.predictor_model(x)
        intrinsic_reward = (target_value - predictor_value).pow(2).sum(1)
        intrinsic_reward = intrinsic_reward.to('cpu').detach().numpy().copy()
        return intrinsic_reward, target_value, predictor_value

    def optimize(self, target_value, predictor_value):
        self.predictor_model.train()
        self.optimizer.zero_grad()
        loss = self.mse_loss(predictor_value, target_value)
        # (.backward) RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # loss.backward()
        self.optimizer.step()
    
    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        intrinsic_reward, target_value, predictor_value = self.get_intrinsic_rewards(x)
        self.optimize(target_value, predictor_value)
        h_v += intrinsic_reward[0]
        return {'policy': h_p, 'value': torch.tanh(h_v)}

# base class of Environment

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.param = args['param'] #env_argsにparamを置いた場合
        self.depth = self.param['depth'] #深度(変更可能)
        self.hyperplane_n = self.param['hyperplane_n'] #超平面次元数(変更可能)
        self.treasure = np.array(self.param['treasure']) #報酬の場所(変更可能) #0番目は必ず(深度-1)になるように
        self.set_reward = self.param['set_reward'] #報酬の値(変更可能)
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

    def Transition(self, action, state):
        """行動、状態を引数に次状態を返す"""
        action = self.action_list_np[action]
        new_state_tmp = [state[i+1]+ action[0][i] for i in range(self.hyperplane_n)]
        new_state = (state[0]+1,) + tuple(new_state_tmp)
        return np.array(new_state) #新しい状態を返す

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
        else: #初期位置を固定にする場合 (False)
            self.state = self.tree_np[0] #[-1, 0], [-1, 0, 0]を最初に入れる

    def play(self, action, player):
        a = np.array([action], )
        next_s = self.Transition(a, self.state)
        self.state = next_s
        if self.pom_bool and np.array_equal(self.state, self.pom_state):
            self.pom_flag = 1

    def terminal(self):
        return self.state[0] == self.depth-1

    def outcome(self):
        # 終端状態に到達した時に報酬を与える関数
        outcomes = [0]
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
                if (self.state == self.treasure).all(axis=1).any(): #treasureが2次元配列じゃないと動かない #(a==b)で同じshapeか，all(axis=1)で列方向に一致しているか，any()でどれか一つにでも当てはまるか，True・Falseを返す
                    treasure_num = np.where((self.treasure == self.state).all(axis=1))[0][0]
                    outcomes = [self.set_reward[treasure_num]]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    def net(self):
        # 切り替え可能
        # return SimpleModel(self.hyperplane_n)
        # return CountBasedModel(self.hyperplane_n)
        return RNDModel(self.hyperplane_n)

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