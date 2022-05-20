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
        self.fc1 = nn.Linear(hyperplane_n + 1, 100) # Environment().hyperplane_n
        # self.fc1 = nn.Linear(Environment().hyperplane_n, 10)
        self.head_p = nn.Linear(100, 2**hyperplane_n) # ここも同様
        self.head_v = nn.Linear(100, 1)
    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        # h = F.relu(self.fc1(x[0][1:]))
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        return {'policy': h_p, 'value': torch.tanh(h_v)}

# base class of Environment

class Environment(BaseEnvironment):
    def __init__(self, args={}):  # args = {'env': 'simpletask', param: {}}
        super().__init__()
        self.param = args['param'] # env_argsにparamを置いた場合
        self.depth = self.param['depth'] #深度(変更可能)
        self.hyperplane_n = self.param['hyperplane_n'] #超平面次元数(変更可能)
        self.treasure = np.array(self.param['treasure']) #報酬の場所(変更可能) #0番目は必ず(深度-1)になるように
        self.set_reward = self.param['set_reward'] #報酬の値(変更可能)
        self.start_random = self.param['start_random'] #初期地点をランダムにするか固定にするか / True: ランダムにする, False: 固定する
        self.pom_bool = self.param['pomdp_setting']['pom_bool'] #POMDPを導入するか / True: 導入する, False: 導入しない
        self.pom_state = self.param['pomdp_setting']['pom_state'] #途中報酬の座標
        self.pom_flag = 0 #途中報酬の座標を通ったら1, 通らなかったら0のまま．0だと報酬が得られない．

        # self.depth = args['depth'] #深度(変更可能)
        # self.hyperplane_n = args['hyperplane_n'] #超平面次元数(変更可能)
        # self.treasure = np.array(args['treasure']) #報酬の場所(変更可能) #0番目は必ず(深度-1)になるように
        # self.set_reward = args['set_reward'] #報酬の値(変更可能)
        # self.start_random = args['start_random'] #初期地点をランダムにするか固定にするか / True: ランダムにする, False: 固定する
        # self.pom_bool = args['pom_bool'] #POMDPを導入するか / True: 導入する, False: 導入しない
        # self.pom_state = args['pom_state'] #途中報酬の座標
        # self.pom_flag = 0 #途中報酬の座標を通ったら1, 通らなかったら0のまま．0だと報酬が得られない．
        self.tree_s = []
        self.action_list = []
        self.tree_make() # state_randomは使えるよ

    def Transition(self, action, state):
        """行動、状態を引数に次状態を返す"""
        ##現在変更を加えている関数
        #この関数のactionにlegal_actionが入力される
        #↓この下の部分でaction_list_npからlegal_actionの番号に対応した行動に変換してる
        #2次元の場合legal_action = 1が選択されていたらactionはaction_list_np[1]ということになる
        #つまり2次元のaction_list[1]は[0 1]になる
        #action = self.action_list_np[action]
        action = self.action_list_np[action]

        # print("state : ",state)
        # print("action : ",action)
        # print("state_depth : ", state)

        #↓ここから主に修正しているところ↓
        #状態遷移がうまくできていないのはactionとnew_state_tmpの配列が原因？
        #print(type(action).__module__ == "numpy")
        #print(action)
        new_state_tmp = [state[i+1]+ action[0][i] for i in range(self.hyperplane_n)]
        new_state = (state[0]+1,) + tuple(new_state_tmp)

        return np.array(new_state) #新しい状態を返す

    def tree_make(self):
        for i in range(2,2+self.depth):
            coord_seed = list(range(0,i))
            coord = list(itertools.product(coord_seed, repeat = self.hyperplane_n))
            coordinate = [(i-2,) + t for t in coord]
            #print(new_p) #各深度の状態
            self.tree_s += coordinate
            if i == 2:
                self.action_list = coord #行動のリスト

        if not self.start_random: # 初期位置固定
            start = (-1,) + (0,) * self.hyperplane_n
            self.tree_s = np.insert(self.tree_s, 0, start, axis=0)

        self.tree_np = np.array(self.tree_s)
        #self.action_list_np = self.action_list
        self.action_list_np = np.array(self.action_list)
        #print("tree_make : ",self.tree_np)
        # print("action_list: ",self.action_list_np)

    def reset(self, args={}): #タスクリセット
        # print(self.action_list)
        if self.start_random: #初期位置をランダムにする場合 (True)
            start = np.random.randint(len(self.action_list_np)) #初期座標ランダム選択    # start: 2*超平面次元 通りある
            self.state = self.tree_np[start] #初期座標にリセット  # [0, x, y], start = 4;
        else: #初期位置を固定にする場合 (False)
            self.state = self.tree_np[0] # [-1, 0], [-1, 0, 0]を最初に入れる
        #print("<><><> start state", self.state)
        # self.a_cnt = 0

    def play(self, action, player):
        a = np.array([action], )
        next_s = self.Transition(a,self.state)
        #print("<><><> action : ",a)
        #print("<><><> new state : ",next_s)
        self.state = next_s
        #self.a_cnt += 1
        if self.pom_bool and np.array_equal(self.state, self.pom_state):
            #print('途中報酬の座標を通りました!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.pom_flag = 1

    # def action_length(self):
    #     return len(self.action_list_np)

    def terminal(self):
        return self.state[0] == self.depth-1

    def outcome(self):
        # terminal outcome
        # 終端状態に到達した時に報酬を与える関数
        outcomes = [-1]
        # if np.array_equal(self.state, self.treasure):
        if self.pom_bool: #True
            if self.pom_flag and (self.state == self.treasure).all(axis=1).any(): #途中報酬の座標を通る && treasureの中にあるか
                outcomes = [1]
            self.pom_flag = 0
        else: #False
            if (self.state == self.treasure).all(axis=1).any(): #treasureが2次元配列じゃないと動かない #(a==b)で同じshapeか，all(axis=1)で列方向に一致しているか，any()でどれか一つにでも当てはまるか，True・Falseを返す
                outcomes = [1]
        # print(self.state)
        #print("<><><><><><> outcomes : ", 'Hit!!!' if outcomes[0]==1 else '--')
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    def net(self):
        return SimpleModel(self.hyperplane_n) # 引数を入れることで解決(多分)

    def observation(self, player=None):
        #if player is None:
            #player = 0
        return self.state.astype(np.float32)
        # return np.zeros(self.hyperplane_n+1).astype(np.float32)

    def legal_actions(self, player):
        #ここでちゃんとアクションを渡せてない
        #エラーが出ない方法
        #ここでアクションの番号を返すように変更中(未完成)
        #1次元の場合[0,1] , 2次元の場合[0,1,2,3] , 3次元の場合[0,1,2,3,4,5,6,7]みたいにする
        #現状は Transition の中身を変更中
        #legal_actions = [x[0] for x in self.action_list_np]
        #legal_actions = np.array([0,1])
        #legal_actions = np.array([0,1,2,3])
        legal_actions = np.arange(2**self.hyperplane_n)
        #legal_actions = np.array([0,1,2,3,4,5,6,7])
        return legal_actions

if __name__ == '__main__':
    e = Environment()
    #print("tree_make",e.tree_np)
    for _ in range(100):
        e.reset() # ここで初期地点をランダムにするか固定にするかする
        while not e.terminal():
            print(e)
            e.play()
        print(e)
        print(e.outcome())
