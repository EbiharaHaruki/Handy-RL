## Config Parameters

This page contains the description of all parameters in `config.yaml`. HandyRL uses yaml-style configuration for training and evaluation.

### Environment Parameters (env_args)

This parameters are used for training and evaluation.

* `env`, type = string
    * environment name
    * **NOTE** default games: TicTacToe, Geister, ParallelTicTacToe, HungryGeese
    * **NOTE** if your environment module is on `handyrl/envs/your_env.py`, set `handyrl.envs.your_env` (split `.py`)


### Training Parameters (train_args)

This parameters are used for training (`python main.py --train`, `python main.py --train-server`).


* `turn_based_training`, type = bool
    * flag for turn-based games (alternating games with multiple players) or not
    * set `True` for alternating turns games (e.g. Tic-Tac-Toe and Geister), `False` for simultaneous games (e.g. HungryGeese)
* `observation`, type = bool
    * whether using opponent features in training
* `return_buckup`, type = bool
    * if learning is done by referring to the next state without using direct returns, set it to `True`. 
    * * **NOTE** Q(λ) does not strictly correspond to multiple Q-learning (hardmax).
    * True: Q-Learning or Actor-Critic
    * False: Policy-Gradient including terminal return
* `gamma`, type = double, constraints: 0.0 <= `gamma` <= 1.0
    * discount rate
* `forward_steps`, type = int
    * steps used to make n-step return estimates for calculating targets of value and advantages of policy
* `compress_steps`, type = int
    * steps to compress episode data for efficient data handling
    * **NOTE** this is a system parameter, so basically no need to change
* `entropy_regularization`, type = double, constraints: `entropy_regularization` >= 0.0
    * coefficient of entropy regularization
* `entropy_regularization_decay`, type = double, constraints: 0.0 <= `entropy_regularization` <= 1.0
    * decay rate of entropy regularization over step progress
    * **NOTE** HandyRL reduces the effect of entropy regularization as the turn progresses
    * **NOTE** larger value decreases the effect, smaller value increases the effect
* `target_model` 
    * `use`, type = True 
        * when using target-net, set it to True. 
        * **NOTE** if you set it to True, be sure to specify update and update_param. 
        * **NOTE** it is not necessary to use it in distributed reinforcement learning if the interval between update_episodes is sufficiently wide.
    * `update`, type = enum
        * `soft` update or  `hard` update
    * `update_param`, type = double
        * In soft update, the weight parameter of the weighted average at the time of target-net update, and in hard update, the update interval of the model parameter.
* `update_episodes`, type = int
    * the interval number of episode to update and save model
    * the models in workers are updated at this timing
* `batch_size`, type = int
    * batch size
* `minimum_episodes`, type = int
    * minimum buffer size to store episode data
    * the training starts after episode data stored more than `minimum_episodes`
* `maximum_episodes`, type = int, constraints: `maximum_episodes` >= `minimum_episodes`
    * maximum buffer size to store episode data
    * the exceeded episode is popped from oldest one
* `epochs`, type = int
    * epochs to stop training
    * **NOTE** If epochs < 0, there is no limit (i.e. keep training)
* `saving_interval_epochs`, type = int
    * epoch interval for saving the model.
    * **NOTE** If epochs <＝ 0, save the model every epoch
* `num_batchers`, type = int
    * the number of batcher that makes batch data in multi-process
* `eval_rate`, type = double, constraints: 0.0 <= `eval_rate` <= 1.0
    * ratio of evaluation worker, the rest is the workers of data generation (self-play)
* `worker`
    * `num_parallel`, type = int
        * the number of worker processes
        * `num_parallel` workers are generated automatically for data generation (self-play) and evaluation
* `lambda`, type = double, constraints: 0.0 <= `lambda` <= 1.0
    * the parameter for lambda values that unifies both Monte Carlo and 1-step TD method
    * **NOTE** Please refer to [TD(λ) wiki](https://en.wikipedia.org/wiki/Temporal_difference_learning#TD-Lambda) for more details.
    * **NOTE** HandyRL computes values using lambda for TD, V-Trace and UPGO
* `policy_target`, type = enum
    * advantage for policy gradient loss
    * `MC`, Monte Carlo
    * `TD`, TD(λ)
    * `TD-Q`, TD(λ) with Q-value (Action Value Function)
    * `TD-Q-HARDMAX`, TD(λ) based on maximum operator with Q-value (Action Value Function)
    * `VTRACE`, [V-Trace described in IMPALA paper](https://arxiv.org/abs/1802.01561)
    * `UPGO`, [UPGO described in AlphaStar paper](https://www.nature.com/articles/s41586-019-1724-z)
* `value_target`, type = enum
    * value target for value loss
    * `MC`, Monte Carlo
    * `TD`, TD(λ)
    * `TD-Q`, TD(λ) with Q-value (Action Value Function)
    * `TD-Q-HARDMAX`, TD(λ) based on maximum operator with Q-value (Action Value Function)
    * `VTRACE`, [V-Trace described in IMPALA paper](https://arxiv.org/abs/1802.01561)
    * `UPGO`, [UPGO described in AlphaStar paper](https://www.nature.com/articles/s41586-019-1724-z)
* `seed`, type = int
    * used to set a seed in learner and workers
    * **NOTE** this seed cannot guarantee the reproducibility for now
* `restart_epoch`, type = int
    * number of epochs to restart training
    * when setting `restart_epoch = 100`, the training restarts from `models/100.pth` and the next model is saved from `models/101.pth`
    * **NOTE** HandyRL checks `models` directory
* `agent`
    * `type`, type = enum
        * agent type and corresponding model type.
        * `BASE`, agent that selects based on the policy distribution
        * `RND`, agent that selects based on the policy distribution, with RND (Random Network Distillation)
        * `QL`, agent that selects based on the Q-value
        * `QL-RND`, agent that selects based on the Q-value, with RND (Random Network Distillation)
            * **NOTE** for Q-value based selection, both `QL` and `QL-RND` require setting a `meta_policy` and its parameters (`param`).
        * `RSRS`, agent that selects based on the RS^2 policy
        * `RSRS-RND`, agent that selects based on the RS^2 policy, with RND (Random Network Distillation)
    * `meta_policy`, type = enum 
        * `greedy`
        * `e-greedy`, epsilon-greedy 
        * `softmax`
    * `param`, type = list
        * if using e-greedy, the parameter becomes the random search probability (epsilon: 1D), and if using softmax, it becomes the temperature parameter (tau: 1D)
        * **NOTE** in the future, it is set as a List type in case of introducing decay algorithms, etc.
    * `metadata`
        * setting of metadata transferred from Learner to Generator at the same interval as the model update 
        * however, unlike the model list, it is not accumulated in the Generator and is updated by replacement every time
        * `name`, type = various
            * list of names of metadata transferred from Learner to Generator
            * **NOTE** Even if the names are set in this list, they will be treated as non-existent if they do not exist.
        * `etc.`
            * the setting of metadata is different for each algorithm, so it is omitted.

### Worker Parameters (worker_args)

This parameters are used only for worker of distributed training (`python main.py --worker`).

* `server_address`, type = string
    * training server address to be connected from worker
    * **NOTE** when training a model on the cloud service (e.g. GCP, AWS), the internal/external IP of virtual machine can be set here
* `num_parallel`, type = int
    * the number of worker processes
    * `num_parallel` workers are generated automatically for data generation (self-play) and evaluation
