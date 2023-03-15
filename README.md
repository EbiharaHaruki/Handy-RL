# HandyRL-SimpleTask
従来の深層強化学習の研究では，アルゴリズムを検証するタスクとしてAtari2600などのデジタルゲームや，強化学習のシミュレーション環境として提供されているOpenAI Gym，囲碁といったボードゲームが用いられてきました．しかし，これらのようなタスクでは細かい難易度調整やアルゴリズムの挙動を詳細に分析することは難しく，計算リソースや学習時間もそれ相応に必要なために簡単な検証に向いていません．

そこで従来のタスクよりも計算リソースを軽量化し，短い時間で学習を進めることができる深層強化学習の検証タスクとして新たにシンプルタスクを開発しました．

<!-- * [HandyRLについて](#HandyRLについて)
* [インストール](#インストール)
* [実行方法](#実行方法) -->


## HandyRLについて
HandyRLの詳細は[ここ](https://github.com/DeNA/HandyRL)から


## インストール方法
まず，HandyRL-SimpleTaskはPython3.7以降でのみ動くので，それ以前のバージョンを使っている場合はアップデートしてください．

最初に，HandyRL-SimpleTaskリポジトリを環境にコピーまたはフォークしてください．また以下のコマンドをターミナル/コマンドプロンプトで実行してください．
```
git clone https://github.com/takalabo/HandyRL-SimpleTask.git
cd HandyRL-SimpleTask
```

次に，追加のライブラリ(NumPyやPyTorchなど)をインストールします．
```
pip3 install -r requirements.txt
```

kaggle環境のゲーム(Hungry Geeseなど)を使用するには，追加でインストールする必要があります．
```
pip3 install -r handyrl/envs/kaggle/requirements.txt
```


## 実行方法

### Step 1: パラメータを設定する
`config.yaml`のパラメータをトレーニングに合わせて以下のように設定します．環境をsimpletask，パラメータをそれぞれ深度8，超平面次元数1，報酬エリア(7, 4)，報酬5で設定し，バッチサイズを64としてトレーニングを実行する場合は，以下のように設定します．

```yaml
env_args:
    env: 'simpletask'
    param:
        depth: 8
        hyperplane_n: 1
        treasure: [[7, 4]]
        set_reward: 5
        ...

train_args:
    ...
    batch_size: 64
    ...
```

注意: HandyRLで実装されている環境の[リスト](handyrl/envs)です．全てのパラメータは，[これ](docs/parameters.md)を参照してください.



### Step 2: トレーニング
パラメータを設定したら，以下のコマンドを実行してトレーニングを開始します．トレーニングされたモデルは，`config.yaml`の`update_episodes`毎に`models`に保存されます．
```
python main.py --train
```


### Step 3: 評価
トレーニング後，任意のモデルに対して評価できます．以下のコマンドは，エポック1のモデルを4プロセスで100ゲーム分評価します．
```
python main.py --eval models/1.pth 100 4
```
注意: デフォルトの対戦相手AIは`evaluation.py`で実装されたランダムなエージェントです．また，自分で任意のエージェントに変更することができます．



### Extra 1: n回平均勝率グラフをプロット
任意の実行回数分データを収集し，平均勝率をプロットします．以下のコマンドは10回分の勝率データを平均してグラフを生成します．
```
. bash_scripts/experiment.sh 10
```

### Extra 2: n回平均報酬グラフをプロット
任意の実行回数分データを収集し，平均報酬をプロットします．以下のコマンドは10回分の報酬データを平均してグラフを生成します．ここで，`config.yaml`のパラメータset_rewardで報酬の値を変更できます．
```
. bash_scripts/experiment_reward.sh 10
```

## アルゴリズムの指定方法
使いたいアルゴリズムに応じて `config.yaml` の該当箇所を変更の所定の箇所を変更する．

応用性が高いゆえに変更箇所が多いので以下にパターンを記載する．

### Policy-Gradieng 系アルゴリズム
以下の特徴を持つアルゴリズムを指定する場合
- Policy の確率分布に基づき行動する
- Value には状態価値関数 V 値のみを持つ
- Value は終端までの方策を利用して学習する
- IS (importance sampling) が適用されるので必然的に Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: False
    forward_steps: 1 以上
    target_model: 
        use: False #（True にしても良いがあまり意味がない）
    policy_target: 'UPGO', 'VTRACE','TD', or 'MC'
    value_target: 'UPGO', 'VTRACE','TD', or 'MC'
    agent: 
        type: 'BASE' or `RND` # RND を利用する場合は後者
        # meta_policy: 記載しない 
    metadata:
        name: [] # 使わない
```

### Q-lerning 系アルゴリズム
以下の特徴を持つアルゴリズムを指定する場合
- 行動価値関数 Q 値に基づき行動する
    - Q 値を使って如何なる方策を作るかは指定する必要がある
- Max oparator を用いて Q 値更新をする Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: True
    forward_steps: 2 # 現在は Q(λ) に対応していないが next state を見る都合上 1 では使えない
    target_model: 
        use: True # 十分 update_episodes が大きければ False でも学習可能
    policy_target: 'TD-Q-HARDMAX'
    value_target: 'TD-Q-HARDMAX'
    agent: 
        type: 'QL' or 'QL-RND' # RND を利用する場合は後者
        meta_policy: 'e-greedy' or 'softmax'
        param: [0.1] # 'e-greedy' ならランダム選択確率，'softmax' なら温度パラメータの数値（list にしているのは今後アルゴリズムが増えた場合を見据えて）
    metadata:
        name: [] # 使わない
```
- システム的なバグを回避するために policy も学習しているが使用はしていない（はず）

### Actor-Critic 系アルゴリズム（未検証）
以下の特徴を持つアルゴリズムを指定する場合
- Policy の確率分布に基づき行動する
- Value には状態価値関数 V 値，のみを用いる
- 学習には軌跡の次状態を利用する
- IS (importance sampling) が適用されるので必然的に Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: True
    forward_steps: 2 以上 # 未検証だが TD(λ) に対応しており next state を見る都合上 1 では使えない
    target_model: 
        use: True # 十分 update_episodes が大きければ False でも学習可能
    policy_target: 'UPGO', 'VTRACE','TD', or 'MC'
    value_target: 'UPGO', 'VTRACE','TD', or 'MC'
    agent: 
        type: 'BASE' or `RND` # RND を利用する場合は後者
        # meta_policy: 記載しない 
    metadata:
        name: []
```

### RS^2 - PG (Policy-Gradieng) 系アルゴリズム
以下の特徴を持つアルゴリズムを指定する場合
- RS^2 から導出される挙動 Policy の確率分布に基づき行動する
- Value には状態価値関数 V 値も行動価値関数 Q 値も用いる
- 終端までの方策を利用して学習する
- IS (importance sampling) が適用されるので必然的に Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: False
    forward_steps: 1 以上
    target_model: 
        use: False #（True にしても良いがあまり意味がない）
    policy_target: 'TD-Q'
    value_target: 'TD-Q'
    agent: 
        type: 'RSRS' or `RSRS-RND` # RND を利用する場合は後者
        # meta_policy: 記載しない 
    metadata:
        # name: ['global_aleph', 'global_return_size'] # K 近傍法を使わない場合
        name: ['knn', 'global_aleph', 'regional_weight', 'global_return_size']
        knn:
            size: 5000 # K 近傍法の memory size
            k: 32 # K 近傍法の近傍としてとる数 K
        regional_weight: 0.5 # knn を使わない場合記載しない（しても動く）
        global_aleph: 1.0 # Global Aspration Level
        global_return_size: 100 # Global Return の保存数
```
- metadata は model 更新（update_episodes の間隔）と同じタイミングで Learner から Generator に転送される
- ただし model と異なり list として Generator に保存されず上書き更新される
- Global Value の更新法は現在ハードコードされている
- Policy は IS のためにしか使用していない

## RS^2 - QL (Q-learning) 系アルゴリズム
以下の特徴を持つアルゴリズムを指定する場合
- RS^2 から導出される挙動 Policy の確率分布に基づき行動する
- Value には状態価値関数 V 値も行動価値関数 Q 値も用いる
- Max oparator を用いて Q 値更新をする Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: True
    forward_steps: 2 # 現在は Q(λ) に対応していないが next state を見る都合上 1 では使えない
    target_model: 
        use: True # 十分 update_episodes が大きければ False でも学習可能
    policy_target: 'TD-Q-HARDMAX'
    value_target: 'TD-Q-HARDMAX'
    agent: 
        type: 'RSRS' or `RSRS-RND` # RND を利用する場合は後者
        # meta_policy: 記載しない 
    metadata:
        # name: ['global_aleph', 'global_return_size'] # K 近傍法を使わない場合
        name: ['knn', 'global_aleph', 'regional_weight', 'global_return_size']
        knn:
            size: 5000 # K 近傍法の memory size
            k: 32 # K 近傍法の近傍としてとる数 K
        regional_weight: 0.5 # knn を使わない場合記載しない（しても動く）
        global_aleph: 1.0 # Global Aspration Level
        global_return_size: 100 # Global Return の保存数
```
- metadata は model 更新（update_episodes の間隔）と同じタイミングで Learner から Generator に転送される
- ただし model と異なり list として Generator に保存されず上書き更新される
- Global Value の更新法は現在ハードコードされている
- システム的なバグを回避するために policy も学習しているが使用はしていない（はず）

## RS^2 - AC (Actor-Critic) 系アルゴリズム（未検証）
以下の特徴を持つアルゴリズムを指定する場合
- RS^2 から導出される挙動 Policy の確率分布に基づき行動する
- Value には状態価値関数 V 値も行動価値関数 Q 値も用いる
- 学習には軌跡の次状態を利用する
- IS (importance sampling) が適用されるので必然的に Off-policy 強化学習
- RND (Random Network Distillation) も併用可能
以下の `config.yaml` の該当箇所を変更
```
    return_buckup: True
    forward_steps: 2 以上 # 未検証だが TD(λ) に対応しており next state を見る都合上 1 では使えない
    target_model: 
        use: True # 十分 update_episodes が大きければ False でも学習可能
    policy_target: 'UPGO', 'VTRACE','TD', or 'MC'
    value_target: 'UPGO', 'VTRACE','TD', or 'MC'
    agent: 
        type: 'RSRS' or `RSRS-RND` # RND を利用する場合は後者
        # meta_policy: 記載しない 
    metadata:
        # name: ['global_aleph', 'global_return_size'] # K 近傍法を使わない場合
        name: ['knn', 'global_aleph', 'regional_weight', 'global_return_size']
        knn:
            size: 5000 # K 近傍法の memory size
            k: 32 # K 近傍法の近傍としてとる数 K
        regional_weight: 0.5 # knn を使わない場合記載しない（しても動く）
        global_aleph: 1.0 # Global Aspration Level
        global_return_size: 100 # Global Return の保存数
```
- metadata は model 更新（update_episodes の間隔）と同じタイミングで Learner から Generator に転送される
- ただし model と異なり list として Generator に保存されず上書き更新される
- Global Value の更新法は現在ハードコードされている
- Policy は IS のためにしか使用していない




<!-- ## ドキュメント

* [**Config Parameters**](docs/parameters.md) shows a list of parameters of `config.yaml`.
* [**Large Scale Training**](docs/large_scale_training.md) is a procedure for large scale training remotely.
* [**Train with Customized Environment**](docs/custom_environment.md) explains an interface of environment to create your own game.
* [**API**](docs/api.md) shows entry-point APIs of `main.py` -->
