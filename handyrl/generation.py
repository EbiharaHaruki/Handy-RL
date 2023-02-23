# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from .agent import agent_class
from .util import softmax
from .metadata import KNN, feed_knn

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
    
    def generate(self, models, metadataset, args):
        # episode generation
        moments = []
        return_metadata = []
        # hidden = {}
        agents = {}
        moment_keys = ['observation', 'selected_prob', 'action_mask', 'action', 'value', 'reward', 'return']
        metadata_keys = []

        if self.env.reset():
            return None
        for player in self.env.players():
            # hidden[player] = models[player].init_hidden() # Reccurent model のための隠れ状態
            agents[player] = agent_class(self.args['agent'])(models[player], metadataset[player], role='g')
            metadata_keys += agents[player].reset(self.env) ## init_hidden() も行われる

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
                if player in turn_players:
                    moment['action'][player] = agents[player].action(self.env, player, action_log=action_log)
                elif player in observers:
                    agents[player].observe(self.env, player, action_log=action_log)
                # print(f'<><><> action: {action}')
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
                moment['reward'][player] = reward.get(player, None)

            # 学習用に保存
            moment['turn'] = turn_players
            moments.append(moment)
            metadata['turn'] = turn_players
            return_metadata.append(metadata)

        if len(moments) < 1:
            return None

        # exec_match ではやっていないこと
        for player in self.env.players():
            ret = 0
            # 各 step に対する割引率付きの target return を step 全体で計算
            for i, m in reversed(list(enumerate(moments))):
                # (m['reward'][player] or 0) は reward が None の対策
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        # episode の詳細情報(moments)は bz2 で圧縮されて送る
        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
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
                # print(f'<><><> feed_knn in generetor.py')
                feed_knn(metadataset[p]['knn'], self.args, [return_metadataset])
                # print(f'<><><> knn.num in generetor.py: {metadataset[p]["knn"].num}')
                break

        return episode, return_metadataset

    def execute(self, models, metadataset, args):
        episode, return_metadata = self.generate(models, metadataset, args)
        if episode is None:
            print('None episode in generation!')
        return episode, return_metadata
