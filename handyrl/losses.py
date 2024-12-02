# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# algorithms and losses

from collections import deque
import torch.nn.functional as F
import torch


def monte_carlo(values, returns):
    return {'target_values': returns, 'advantages': returns - values}


def temporal_difference(values, returns, rewards, teaminals, bonuses, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        target_values.appendleft(reward + bonus + not_teaminals * gamma * ((1 - lmb) * values[:, i + 1] + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return {'target_values': target_values, 'advantages': target_values - values}


def temporal_difference_q(values, qvalues, returns, rewards, teaminals, bonuses, lmb, gamma):
    _rewards = rewards if rewards is not None else 0
    _bonuses = bonuses if bonuses is not None else 0
    # 終端 return から
    # multi-step reward は別の場所で計算
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        # 計算効率の面からも target_values を後ろから計算していく
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        # TD(0), λ = 0 なら 1 step buckup のみをする
        # TD(0), λ = 0 なら割引率を累乗された return のみを使う 
        # target_values[0] は appendleft されていくので常に一つ後の状態の価値になる
        target_values.appendleft(reward + bonus + not_teaminals * gamma * ((1 - lmb) * values[:, i + 1] + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)
    target_values_t_plus_1 = torch.cat([target_values[:, 1:], returns[:, -1:]], dim=1)
    target_q_value = _rewards + _bonuses + gamma * target_values_t_plus_1
    advantages = target_q_value - values

    return {
        'target_values': target_values, 
        'advantages': advantages, 
        'qvalue': target_q_value, 
        'target_advantage_for_q': advantages
        }


def temporal_difference_q_hardmax(values, qvalues, target_qvalues, returns, rewards, teaminals, bonuses, lmb, gamma):
    # 終端 return から
    # multi-step reward は別の場所で計算
    target_values = deque([returns[:, -1]])
    if qvalues is not None:
        best_actions = torch.argmax(qvalues, dim=-1, keepdim=True)
        max_qvalues = target_qvalues.gather(-1,best_actions)
    else:
        max_qvalues = 0
    for i in range(values.size(1) - 2, -1, -1):
        # 計算効率の面からも target_values を後ろから計算していく
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        # TD(0), λ = 0 なら 1 step buckup のみをする
        # 現在は基本 λ = 0 の運用を想定
        target_values.appendleft(reward + bonus + not_teaminals * gamma * ((1 - lmb) * max_qvalues[:, i + 1] + lmb * target_values[0]))
    target_values = torch.stack(tuple(target_values), dim=1)

    return {
        'target_values': target_values, 
        'advantages': target_values - values, 
        'qvalue': target_values, 
        'target_advantage_for_q': target_values - values
        }


def upgo(values, returns, rewards, teaminals, bonuses, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        value = values[:, i + 1]
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        # 1 step buckup と TD(λ) buckup から高い方を使う
        target_values.appendleft(reward + bonus + not_teaminals * gamma * torch.max(value, (1 - lmb) * value + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return {'target_values': target_values, 'advantages': target_values - values}


def vtrace(values, returns, rewards, teaminals, bonuses, lmb, gamma, rhos, cs):
    rewards = rewards if rewards is not None else 0
    bonuses = bonuses if bonuses is not None else 0
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1) # value of next obs
    deltas = rhos * (rewards + bonuses + gamma * values_t_plus_1 - values) # TD-error

    # compute Vtrace value target recursively
    multi_step_deltas = deque([deltas[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        multi_step_deltas.appendleft(deltas[:, i] + not_teaminals * gamma * lmb * cs[:, i] * multi_step_deltas[0])

    multi_step_deltas = torch.stack(tuple(multi_step_deltas), dim=1)
    target_values = multi_step_deltas + values
    target_values_t_plus_1 = torch.cat([target_values[:, 1:], returns[:, -1:]], dim=1)
    target_q_value = rewards + bonuses + gamma * target_values_t_plus_1
    #advantages = rewards + bonuses + gamma * target_values_t_plus_1 - values
    advantages = target_q_value - values

    return {'target_values': target_values,
            'advantages': advantages,
            'qvalue': target_q_value, 
            'target_advantage_for_q': advantages}

def sac(values, q1, q2, log,  returns, rewards, teaminals, bonuses, lmb, gamma):
    alpha = 0.2
    _rewards = rewards if rewards is not None else 0
    _bonuses = bonuses if bonuses is not None else 0
    # 終端 return から
    # multi-step reward は別の場所で計算
    target_values = deque([returns[:, -1]])
    #print(f'q1 = {q1},q2 = {q2},log = {log}')
    for i in range(values.size(1) - 2, -1, -1):
        # 計算効率の面からも target_values を後ろから計算していく
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        # TD(0), λ = 0 なら 1 step buckup のみをする
        # TD(0), λ = 0 なら割引率を累乗された return のみを使う 
        # target_values[0] は appendleft されていくので常に一つ後の状態の価値になる
        
        
        #action_distribution = torch.distributions.Categorical(prob)
        #action = action_distribution.sample()
        #log_prob = action_distribution.log_prob(action)
        #target_q = reward + (1 - lmb) * gamma * v
        target_values.appendleft(reward + bonus + not_teaminals * gamma * ((1 - lmb) * values[:, i + 1] + lmb * target_values[0]))
        #target_values.appendleft(reward + bonus + not_teaminals * gamma * (target_q))
        #print(f'{target_values}')

    prob = F.softmax(log, dim=-1)
    log_prob = F.log_softmax(prob,dim=-1)
    min_q = torch.min(q1,q2)
    v = (prob * (min_q - alpha * log_prob)).sum(dim=-1, keepdim=True)
    #print(f"reward shape: {rewards.shape}")
    #print(f"value shape: {v.shape}")
    target_q = rewards + (1 - lmb) * gamma * v
    #print(f"targetq = {target_q}")
    target_values = torch.stack(tuple(target_values), dim=1)
    #print(f"targetv = {target_values}")
    advantages = target_values - values
    

    return {
        'target_values': target_q, 
        'advantages': 0, 
        }


def compute_rnd(embed_state, embed_state_fix):
    return (embed_state - embed_state_fix) ** 2


def compute_target(algorithm, values, returns, rewards, teaminals, lmb, gamma, rhos, cs, qvalues=None, q1=None,q2=None,log=None,target_qvalues=None, bonuses=None):
    if values is None:
        # In the absence of a baseline, Monte Carlo returns are used.
        return {'target_values': returns, 'advantages': returns}

    if algorithm == 'MC':
        return monte_carlo(values, returns)
    elif algorithm == 'TD':
        return temporal_difference(values, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'TD-Q':
        return temporal_difference_q(values, target_qvalues, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'TD-Q-HARDMAX':
        return temporal_difference_q_hardmax(values, qvalues, target_qvalues, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'UPGO':
        return upgo(values, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'VTRACE':
        return vtrace(values, returns, rewards, teaminals, bonuses, lmb, gamma, rhos, cs)
    elif algorithm == 'SAC':
        return sac(values, q1, q2, log, returns, rewards, teaminals, bonuses, lmb, gamma)
    else:
        print('No algorithm named %s' % algorithm)
