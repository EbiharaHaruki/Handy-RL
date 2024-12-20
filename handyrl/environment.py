# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# game environment

import importlib


ENVS = {
    'TicTacToe':         'handyrl.envs.tictactoe',
    'Geister':           'handyrl.envs.geister',
    'ParallelTicTacToe': 'handyrl.envs.parallel_tictactoe',
    'HungryGeese':       'handyrl.envs.kaggle.hungry_geese',
    'simpletask':        'handyrl.envs.simpletask',
    'simple_pyramid':    'handyrl.envs.simple_pyramid',
    'CartPole':          'handyrl.envs.gym.cart_pole',
}


def prepare_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    elif hasattr(env_module, 'prepare'):
        env_module.prepare()


def make_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    else:
        return env_module.Environment(env_args)


# base class of Environment

class BaseEnvironment:
    def __init__(self, args={}):
        pass

    def __str__(self):
        return ''

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        raise NotImplementedError()

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        raise NotImplementedError()

    #
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [self.turn()]

    #
    # Should be defined if there are other players besides the turn player
    # who should observe the environment (mainly with RNNs)
    #
    def observers(self):
        return []

    #
    # Should be defined in all games
    #
    def terminal(self):
        raise NotImplementedError()

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {}

    #
    # Should be defined in all games
    #
    def outcome(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you encode action as special string
    #
    def action2str(self, a, player=None):
        return str(a)

    #
    # Should be defined if you encode action as special string
    #
    def str2action(self, s, player=None):
        return int(s)

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self, player=None):
        return ''

    #
    # Should be defined if you use network battle mode
    #
    def update(self, info, reset):
        raise NotImplementedError()

    #
    # Should be defined if you want to make the environment non-stationary
    #
    def shift_env(self, num_episodes):
        raise False

    #
    # Should be defined if you want to display the environment's status
    #
    def fprint_env_status(self, role, worker_id):
        return False

    #
    # Should be defined if you want to retrieve the seed set for the environment
    #
    def get_seed(self):
        return {}