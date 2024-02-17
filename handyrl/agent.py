# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import sys
import random
import faiss

import numpy as np

from .util import softmax

def agent_class(args):
    if args['type'] == 'BASE' or args['type'] == 'RND':
        return Agent
    elif args['type'] == 'QL' or args['type'] == 'QL-RND':
        return QAgent
    elif args['type'] == 'RSRS' or args['type'] == 'RSRS-RND':
        return RSRSAgent
    elif args['type'] == 'R4D-RSRS' or args['type'] == 'R4D-RSRS-RND':
        return R4DRSRSAgent
    # elif args['type'] == 'SRS' or args['type'] == 'SRS-RND':
    #     return SRSAgent
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
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True, args=None):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.metadataset = metadataset
        self.hidden = None
        self.role = role
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


def greedy_from_value(actions, value, dummy, generating=True):
    ap_list = sorted([(a, value[a]) for a in actions], key=lambda x: -x[1])
    action = ap_list[0][0]
    selected_prob = 1.0
    return action, selected_prob

def e_greedy_from_value(actions, value, epsilon, generating=True):
    e = epsilon[0] if generating else 0.0
    if e > np.random.rand():
        action = random.choices(np.arange(len(actions)))[0]
        selected_prob = e/len(actions)
    else:
        ap_list = sorted([(a, value[a]) for a in actions], key=lambda x: -x[1])
        action = ap_list[0][0]
        selected_prob = 1.0 - e + e/len(actions)
    return action, selected_prob

def softmax_from_value(actions, value, temperature, generating=True):
    t = temperature[0]
    if generating or t != 0.0:
        policy = softmax(value) if t is None else softmax(value / t)
        action = random.choices(np.arange(len(actions)), weights=policy)[0]
        selected_prob = policy[action]
    else:
        ap_list = sorted([(a, value[a]) for a in actions], key=lambda x: -x[1])
        action = ap_list[0][0]
        selected_prob = 1.0
    return action, selected_prob

class QAgent(Agent):
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True, args={'meta_policy': None}):
        super().__init__(model, metadataset, role, temperature, observation, args)
        if args['meta_policy'] == 'e-greedy':
            self.meta_policy = e_greedy_from_value
        elif args['meta_policy'] == 'softmax':
            self.meta_policy = softmax_from_value
        else:
            self.meta_policy = greedy_from_value
        self.param = args.get('param', None)
        if self.param is None and not self.generating:
            self.param = [0.0]

    def action(self, env, player, show=False, action_log=None):
        # learning = (action_log is not None)
        obs = env.observation(player)
        outputs = self.plan(obs) # reccurent model 対応
        actions = env.legal_actions(player)
        v = outputs.get('value', None)
        p = outputs['policy']
        try:
            q = outputs['qvalue']
        except KeyError:
            sys.exit("SystemExit: Q value is None")
        action_mask = np.ones_like(p) * 1e32
        action_mask[actions] = 0
        q = q - action_mask

        # ap_list = sorted([(a, q[a]) for a in actions], key=lambda x: -x[1])
        # action = ap_list[0][0]
        # selected_prob = 1.0
        action, selected_prob = self.meta_policy(actions, q, self.param, self.generating)

        if show:
            print_outputs(env, softmax(p), v)

        # action log は action 決定過程の情報
        if self.generating:
            action_log['moment']['observation'][player] = obs
            action_log['moment']['value'][player] = v
            action_log['moment']['qvalue'][player] = q
            action_log['moment']['selected_prob'][player] = selected_prob
            action_log['moment']['action_mask'][player] = action_mask

        return action


class RSRSAgent(Agent):
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True, args=None):
        super().__init__(model, metadataset, role, temperature, observation, args)
        self.metadata_keys = ['latent', 'action']
        # TODO なぜか偶に metadata の key が player になってる問題解決
        self.global_aleph = metadataset.get('global_aleph', 1.0)
        self.rw = metadataset.get('regional_weight', 0.0)

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()
        return self.metadata_keys

    def action(self, env, player, show=False, action_log=None):
        # learning = (action_log is not None)
        global_aleph = self.global_aleph
        if action_log is not None:
            global_v = action_log['global_v'][player]
            global_delta = np.amax([global_aleph - global_v, 0.0])
        else:
            global_v = None
            global_delta = 0.0
        obs = env.observation(player)
        outputs = self.plan(obs) # reccurent model 対応
        actions = env.legal_actions(player)
        v = outputs.get('value', None)
        p_nn = outputs['policy']
        q = outputs.get('qvalue', None)
        aleph = global_delta + np.amax(q) if q is not None else 0.0
        c_nn = softmax(outputs.get('confidence', None).squeeze())
        latent = outputs['latent']
        if 'knn' in self.metadataset:
            c_reg = self.metadataset['knn'].regional_nn(latent)
            if c_reg is not None:
                c = self.rw * c_reg.squeeze() + (1.0 - self.rw) * c_nn
            else:
                p_reg = 1.0 / actions.size
                c_reg = np.full(actions.size, p_reg)
                c = c_nn
        else:
            p_reg = 1.0 / actions.size
            c_reg = np.full(actions.size, p_reg)
            c = c_nn

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
        p = np.log(rsrs)
        action_mask = np.ones_like(p) * 1e32
        action_mask[actions] = 0
        p = p - action_mask
        p_srs = softmax(p)
        entropy_srs = -(p_srs * np.ma.log(p_srs)/np.ma.log(len(p))).sum()

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
            action_log['moment']['c'][player] = c
            action_log['moment']['c_reg'][player] = c_reg
            # action_log['metadata']['entropy_srs'][player] = entropy_srs

        return action

class R4DRSRSAgent(Agent):
    def __init__(self, model, metadataset, role='e', temperature=None, observation=True, args=None):
        super().__init__(model, metadataset, role, temperature, observation, args)
        self.metadata_keys = ['latent', 'action']
        self.global_aleph = metadataset.get('global_aleph', 1.0)
        self.rw = metadataset.get('regional_weight', 0.0)

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()
        return self.metadata_keys

    def action(self, env, player, show=False, action_log=None):
        # learning = (action_log is not None)
        global_aleph = self.global_aleph
        if action_log is not None:
            global_v = action_log['global_v'][player]
            global_delta = np.amax([global_aleph - global_v, 0.0])
        else:
            global_v = None
            global_delta = 0.0
        obs = env.observation(player)
        outputs = self.plan(obs) # reccurent model 対応
        actions = env.legal_actions(player)
        v = outputs.get('value', None)
        p_nn = outputs['policy']
        q = outputs.get('qvalue', None)
        aleph = global_delta + np.amax(q) if q is not None else 0.0
        c_predict = outputs.get('confidence_57', None)
        c_target = outputs.get('confidence_57_fix', None)

        # 分割
        c_predict = c_predict.reshape(-1, actions.size)
        c_target = c_target.reshape(-1, actions.size)
        # 計算
        rnd = np.mean((c_target - c_predict)**2,axis=0)
        rnd = 1e-6/(rnd+1e-6)
        #正規化
        c_nn = rnd/np.sum(rnd)

        latent = outputs['latent']
        if 'knn' in self.metadataset:
            c_reg = self.metadataset['knn'].regional_nn(latent)
            if c_reg is not None:
                c = self.rw * c_reg.squeeze() + (1.0 - self.rw) * c_nn
            else:
                p_reg = 1.0/actions.size
                c_reg = np.full(actions.size, p_reg)
                c = c_nn
        else:
            p_reg = 1.0/actions.size
            c_reg = np.full(actions.size, p_reg)
            c = c_nn

        # rs = c * (q - aleph)
        if np.amax(q) >= aleph:
            #print("満足")
            fix_aleph = np.amax(q) + sys.float_info.epsilon
            delta = fix_aleph - q
            if np.amin(delta) < 0:
                delta -= np.amin(delta)  # 丸め誤差が発生した場合の処理
            if np.any(delta == 0.0):
                delta += sys.float_info.epsilon
        else:
            #print("非満足")
            delta = aleph - q
        z = 1.0 / np.sum(1.0 / delta)
        rho = z / delta #17
        rsrs = (np.max(c / rho) + sys.float_info.epsilon) * rho - c
        if np.min(rsrs) < 0:
            rsrs -= np.min(rsrs)
        if np.any(rsrs == 0.0):
            rsrs += sys.float_info.epsilon
        p = np.log(rsrs)
        action_mask = np.ones_like(p) * 1e32
        action_mask[actions] = 0
        p = p - action_mask
        p_srs = softmax(p)
        entropy_srs = -(p_srs * np.ma.log(p_srs)/np.ma.log(len(p))).sum()

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
            action_log['moment']['c'][player] = c
            action_log['moment']['c_reg'][player] = c_reg
            # action_log['metadata']['entropy_srs'][player] = entropy_srs

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
