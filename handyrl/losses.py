# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# algorithms and losses

from collections import deque

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


def temporal_difference_q(values, qavlues, returns, rewards, teaminals, bonuses, lmb, gamma):
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

    return {
        'target_values': target_values, 
        'advantages': target_values - values, 
        'qvalue': target_values, 
        'target_advantage_for_q': target_values - values
        }


def temporal_difference_q_hardmax(values, qavlues, returns, rewards, teaminals, bonuses, lmb, gamma):
    # 終端 return から
    # multi-step reward は別の場所で計算
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        # 計算効率の面からも target_values を後ろから計算していく
        reward = rewards[:, i] if rewards is not None else 0
        bonus = bonuses[:, i] if bonuses is not None else 0
        max_qavlues = (torch.max(qavlues, dim=-1, keepdim=True)).values if qavlues is not None else 0
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        # TD(0), λ = 0 なら 1 step buckup のみをする
        # 現在は基本 λ = 0 の運用を想定
        target_values.appendleft(reward + bonus + not_teaminals * gamma * ((1 - lmb) * max_qavlues[:, i + 1] + lmb * target_values[0]))

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


def vtrace(values, returns, rewards, teaminals, lmb, gamma, rhos, cs):
    rewards = rewards if rewards is not None else 0
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1)
    deltas = rhos * (rewards + gamma * values_t_plus_1 - values)

    # compute Vtrace value target recursively
    vs_minus_v_xs = deque([deltas[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        not_teaminals = (1.0 - teaminals[:, i+1]) if teaminals is not None else 1
        vs_minus_v_xs.appendleft(deltas[:, i] + not_teaminals * gamma * lmb * cs[:, i] * vs_minus_v_xs[0])

    vs_minus_v_xs = torch.stack(tuple(vs_minus_v_xs), dim=1)
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[:, 1:], returns[:, -1:]], dim=1)
    advantages = rewards + gamma * vs_t_plus_1 - values

    return {'target_values': vs, 'advantages': advantages}


def compute_rnd(embed_state, embed_state_fix):
    return (embed_state - embed_state_fix) ** 2


def compute_target(algorithm, values, returns, rewards, teaminals, lmb, gamma, rhos, cs, qavlues=None, bonuses=None):
    if values is None:
        # In the absence of a baseline, Monte Carlo returns are used.
        return {'target_values': returns, 'advantages': returns}

    if algorithm == 'MC':
        return monte_carlo(values, returns)
    elif algorithm == 'TD':
        return temporal_difference(values, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'TD-Q':
        return temporal_difference_q(values, qavlues, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'TD-Q-HARDMAX':
        return temporal_difference_q_hardmax(values, qavlues, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'UPGO':
        return upgo(values, returns, rewards, teaminals, bonuses, lmb, gamma)
    elif algorithm == 'VTRACE':
        return vtrace(values, returns, rewards, teaminals, lmb, gamma, rhos, cs)
    else:
        print('No algorithm named %s' % algorithm)
