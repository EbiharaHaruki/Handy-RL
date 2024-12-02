# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle
import copy

import numpy as np

from .agent import agent_class
from .util import softmax
from .metadata import KNN, feed_knn

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        # RS 関係の処理
        if 'global_aleph' in args['metadata']['name']:
            self.global_v = {}
            self.global_n = {}
            self.global_returns = {}
            self.lastidx = {}
            self.num = {}
            self.size = args['metadata']['global_return_size']
            for player in self.env.players():
                self.global_v[player] = 0.0
                self.global_returns[player] = np.zeros(self.size, dtype=float)
                self.lastidx[player] = 0
                self.num[player] = 0

    def generate(self, models, metadataset, args):
        # episode generation
        moments = []
        return_metadata = []
        # hidden = {}
        agents = {}
        moment_keys = ['observation', 'selected_prob', 'action_mask', 'action', 'value', 'qvalue', 'reward', 'return', 'terminal', 'c', 'c_reg','c_nn','entropy_srs', 'c_accuracy', 'state_index']
        metadata_keys = []

        if self.env.reset():
            return None
        for player in self.env.players():
            # hidden[player] = models[player].init_hidden() # Reccurent model のための隠れ状態
            self.args['agent']['play_subagent_prob'] = args['play_subagent_prob']
            agents[player] = agent_class(self.args['agent'])(models[player], metadataset[player], role='g', args=self.args['agent'])
            metadata_keys += agents[player].reset(self.env) ## init_hidden() も行われる
            if hasattr(self, 'global_v'):
                metadataset[player]['global_v'] = self.global_v[player]

        while not self.env.terminal():
            # 'observation': 観測（状態系列）
            # 'selected_prob': 挙動方策（IS 用）
            # 'action_mask': action mask
            # 'action': action
            # 'value': value 推定値
            # 'reward': 報酬
            # 'return': 割引率付き収益
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}
            metadata = {key: {p: None for p in self.env.players()} for key in metadata_keys}

            # 誰のターンか決める
            turn_players = self.env.turns()
            # player ではない observers の lsit
            observers = self.env.observers()
            # player ごとのターン経過
            for player in self.env.players():
                if player not in turn_players + observers:
                    continue
                if player not in turn_players and player in args['player'] and not self.args['observation']:
                    continue

                # player model wrapper
                # model = models[player]
                #begin# Agent.action()
                action_log = {'moment': moment, 'metadata': metadata}
                if hasattr(self, 'global_v'):
                    action_log['global_v'] = self.global_v
                if player in turn_players:
                    moment['action'][player] = agents[player].action(self.env, player, action_log=action_log)
                elif player in observers:
                    agents[player].observe(self.env, player, action_log=action_log)
                # # 観測取得
                # obs = self.env.observation(player)
                # #begin# Agent.plan()
                # # Reccurent 情報 (hidden) と共に評価
                # outputs = model.inference(obs, hidden[player])
                # # 内部状態保存
                # hidden[player] = outputs.get('hidden', None)
                # #end# Agent.plan()
                # v = outputs.get('value', None)
                # #end# Agent.action()

                # # 学習用に保存
                # moment['observation'][player] = obs
                # moment['value'][player] = v

                # # exec_match でやっていること
                # if player in turn_players:
                #     #begin# Agent.action
                #     p_ = outputs['policy']
                #     legal_actions = self.env.legal_actions(player)
                #     action_mask = np.ones_like(p_) * 1e32
                #     action_mask[legal_actions] = 0
                #     # softmax をここで既に掛けている
                #     p = softmax(p_ - action_mask)
                #     # 温度パラメータ temperature がない
                #     action = random.choices(legal_actions, weights=p[legal_actions])[0]
                #     #end# Agent.action
                #     # 学習用に保存
                #     moment['selected_prob'][player] = p[action]
                #     moment['action_mask'][player] = action_mask
                #     moment['action'][player] = action

            # exec_match でやっていること
            if self.env.step(moment['action']):
                return None

            # exec_match ではやっていないこと
            reward = self.env.reward()
            for player in self.env.players():
                # 学習用に保存
                # moment['reward'][player] = reward.get(player, None)
                # outcome を reward と同等に扱うため存在しないなら None ではなく 0.0 にする
                moment['reward'][player] = reward.get(player, 0)
                moment['terminal'][player] = 0 # 1 if self.env.terminal() else 0

            # 学習用に保存
            moment['turn'] = turn_players
            moments.append(moment)
            metadata['turn'] = turn_players
            return_metadata.append(metadata)

        len_episode = len(moments)
        if len(moments) < 1:
            return None
        
        # forward_steps で終端まで見なかったり MC 法を使わない場合には終端の buck up 用のダミー状態を入力
        # Q 学習等のためには終端の後にもう一つ state などの情報を格納する必要がある
        # post terminal state(summy)
        if self.args['return_buckup']:
            last_moment = copy.deepcopy(moments[len_episode-1])
            last_moment['reward'][player] = 0.0 # reward を 0 に
            last_moment['terminal'][player] = 1 # dummy
            moments.append(last_moment)

        outcome = self.env.outcome()

        # exec_match ではやっていないこと
        for player in self.env.players():
            # outcome を reward と同等に扱うため代入
            moments[len_episode-1]['reward'][player] = outcome[player]
            ret = 0.0  # 終端 return から入れる
            g_ret = 0.0
            # 各 step に対する割引率付きの target return を step 全体で計算
            for i, m in reversed(list(enumerate(moments))):
                # (m['reward'][player] or 0) は reward が None の対策
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret #収益
                g_ret = (m['reward'][player] or 0) + 1.0 * g_ret #割引率なし収益
                moments[i]['return'][player] = ret
            if hasattr(self, 'global_returns'):
                l = self.lastidx[player]
                self.global_returns[player][l] = g_ret #エピソードの長さぶん格納
                self.lastidx[player] = (self.lastidx[player] + 1) if l >= (self.size - 1) else 0
                if self.num[player] == self.size:
                    self.num[player] += 1
                    #self.global_v[player] = self.global_returns[player][0:self.num[player]].mean()
                    self.global_v[player] = np.max(self.global_returns[player][0:self.num[player]])
                else:
                    #self.global_v[player] = self.global_returns[player].mean()
                    self.global_v[player] = np.amax(self.global_returns[player])

        # episode の詳細情報(moments)は bz2 で圧縮されて送る
        episode = {
            'args': args, 'steps': len(moments),
            'outcome': outcome,
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }
        return_metadataset = {
            'args': args, 'steps': len(return_metadata),
            'metadata': return_metadata
        }
        for p in metadataset:
            if 'knn' in metadataset[p]:
                # 共有しているので player 0 のみ
                feed_knn(metadataset[p]['knn'], self.args, [return_metadataset])
                break

        return episode, return_metadataset

    def execute(self, models, metadataset, args):
        episode, return_metadata = self.generate(models, metadataset, args)
        if episode is None:
            print('None episode in generation!')
        return episode, return_metadata
