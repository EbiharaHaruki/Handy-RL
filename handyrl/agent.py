# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import sys
import random
import faiss

import numpy as np

from .util import softmax

def agent_class(args):
    if args['type'] == 'BASE':
        return Agent
    elif args['type'] == 'RS':
        return RSAgent
    else:
        print('No agent named %s' % args['agent'])

class RandomAgent:
    def reset(self, env, show=False):
        return []

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player, key=self.key)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.metadataset = metadataset
        self.hidden = None
        self.generating = (role == 'g')
        if (temperature is None) and (not self.generating): 
            # eval 中かつ 温度 param が入力されていない時
            self.temperature = 0.0
        else:
            self.temperature = temperature # None or (not 0)
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()
        return []

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False, action_log=None):
        # learning = (action_log is not None)
        obs = env.observation(player)
        outputs = self.plan(obs) # reccurent model 対応
        actions = env.legal_actions(player)
        v = outputs.get('value', None)
        p = outputs['policy']
        action_mask = np.ones_like(p) * 1e32
        action_mask[actions] = 0
        p = p - action_mask

        if self.generating or self.temperature != 0.0:
            policy = softmax(p) if self.temperature is None else softmax(p / self.temperature)
            action = random.choices(np.arange(len(p)), weights=policy)[0]
            selected_prob = policy[action]
            if show:
                print_outputs(env, policy, v)
        else:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            action = ap_list[0][0]
            selected_prob = 1.0
            if show:
                print_outputs(env, softmax(p), v)

        # action log は action 決定過程の情報
        if self.generating:
            action_log['moment']['observation'][player] = obs
            action_log['moment']['value'][player] = v
            action_log['moment']['selected_prob'][player] = selected_prob
            action_log['moment']['action_mask'][player] = action_mask

        return action

    def observe(self, env, player, show=False, action_log=None):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
            if self.generating:
                action_log['moment']['observation'][player] = obs
                action_log['moment']['value'][player] = v
        return v

class RSAgent(Agent):
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True):
        super().__init__(model, metadataset, role, temperature, observation)
        self.metadata_keys = ['latent', 'action']
        # TODO なぜか偶に metadata の key が player になってる問題解決
        self.grobal_aleph = metadataset.get('grobal_aleph', 1.0)
        self.rw = metadataset.get('regional_weight', 0.0)

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()
        return self.metadata_keys

    def action(self, env, player, show=False, action_log=None):
        # learning = (action_log is not None)
        aleph = self.grobal_aleph
        obs = env.observation(player)
        outputs = self.plan(obs) # reccurent model 対応
        actions = env.legal_actions(player)
        v = outputs.get('value', None)
        p_nn = outputs['policy']
        q = outputs.get('qvalue', None)
        c_nn = outputs.get('confidence', None)
        latent = outputs['latent']
        if 'knn' in self.metadataset:
            c_reg = self.metadataset['knn'].regional_nn(latent)
        else:
            c_reg = None
        c = c_nn if c_reg is None else self.rw * c_reg.squeeze() + (1.0 - self.rw) * c_nn

        # rs = c * (q - aleph)
        if np.amax(q) >= aleph:
            fix_aleph = np.amax(q) + sys.float_info.epsilon
            delta = fix_aleph - q
            if np.amin(delta) < 0:
                delta -= np.amin(delta)  # 丸め誤差が発生した場合の処理
            if np.any(delta == 0.0):
                delta += sys.float_info.epsilon
        else:
            delta = aleph - q
        z = 1.0 / np.sum(1.0 / delta)
        rho = z / delta
        rsrs = (np.max(c / rho) + sys.float_info.epsilon) * rho - c
        if np.min(rsrs) < 0:
            rsrs -= np.min(rsrs)
        if np.any(rsrs == 0.0):
            rsrs += sys.float_info.epsilon
        p = rsrs
        # print(f'<><><> p_nn: {p_nn}')
        # print(f'<><><> p_rsrs: {p}')
        # print(f'<><><> action_mask: {np.ones_like(p)}')
        # print(f'<><><> p_rsrs: {p/p.sum()}')
        action_mask = np.ones_like(p) * 1e32
        action_mask[actions] = 0
        p = p - action_mask

        if self.generating or self.temperature != 0.0:
            policy = softmax(p) if self.temperature is None else softmax(p / self.temperature)
            action = random.choices(np.arange(len(p)), weights=policy)[0]
            selected_prob = policy[action]
            if show:
                print_outputs(env, policy, v)
        else:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            action = ap_list[0][0]
            selected_prob = 1.0
            if show:
                print_outputs(env, softmax(p), v)
        one_hot_action = np.identity(len(p))[action]

        # action log は action 決定過程の情報
        if self.generating:
            action_log['moment']['observation'][player] = obs
            action_log['moment']['value'][player] = v
            # TODO 満足していると 0.0 が入っちゃう問題
            action_log['moment']['selected_prob'][player] = selected_prob
            action_log['moment']['action_mask'][player] = action_mask
            action_log['metadata']['latent'][player] = latent
            action_log['metadata']['action'][player] = one_hot_action

        return action


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o.items():
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [v]
        for k, vl in outputs.items():
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model):
        super().__init__(model, temperature=1.0)
