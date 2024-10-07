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

#### 注意
- デフォルトの対戦相手AIは`evaluation.py`で実装されたランダムなエージェントです．また，自分で任意のエージェントに変更することができます．
- 環境の状態特徴量が実行の度ランダムに設定される (e.g. `simple_pyramid.py`) 場合，feature に関する seed を学習時と合わせないと正しく評価できません．
    - 報酬や初期状態，観測ノイズ等の乱数 seed はもちろん合わせなくて OK


### Extra 1: n回平均報酬グラフをプロット
任意の実行回数分データを `trainlog` 内に日付名のディレクトリを生成して収集し，平均報酬をプロットします．以下のコマンドは `simple_pyramid.py` 環境を 10回分の報酬データを平均してグラフを生成します．

#### 注意
- `config.yaml`の `env_args` の `env` パラメータで決まる環境名と以下 bash コマンドの環境名は同じである必要があります（違ってもエラーは出ない）
- bash 環境名はグラフの出力時のタイトルになります．
- 実行時に `環境名.py` と `config.yaml` が日付名ディレクトリにコピーされ保存されます
    - ハイパーパラメータとモデル構造を保存するため
    - gym 環境は `gym/環境名` で指定する必要あり

``` 
# simple_pyramid の場合
. bash_scripts/experiment_reward.sh 10 simple_pyramid

# gym/cart_pole の場合
. bash_scripts/experiment_reward.sh 10 cart_pole
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
        type: 'BASE'
        use_RND: True # RND を利用する場合
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
        type: 'QL'
        use_RND: True # RND を利用する場合
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
        type: 'BASE'
        use_RND: True # RND を利用する場合は後者
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
        type: 'RSRS'
        use_RND: True # RND を利用する場合は後者
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
        type: 'RSRS'
        use_RND: True # RND を利用する場合
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
        type: 'RSRS'
        use_RND: True # RND を利用する場合
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


## A-S-C 系アルゴリズム
以下の特徴を持つアルゴリズムを指定する場合
- 方策の表現学習 (A-S-C) を行う
- `subtype` で指定された行動で軌跡生成する（設定値はそれらのエージェント参照）
- RND (Random Network Distillation) も併用可能

以下の `config.yaml` の該当箇所を変更
その他のパラメータは `subtype` エージェントの設定に依存する
```
    agent: 
        type: 'A-S-C'
        use_RND: True # RND を利用する場合
        subtype: 任意のエージェント
        play_subagent_base_prob: 1.0 # サブエージェントが軌跡を生成する確率の初期値
        play_subagent_lower_prob: 0.5 # サブエージェントが軌跡を生成する確率の下限
        play_subagent_decay_per_ep: 0.000001 # サブエージェントが軌跡を生成する確率のエピソードごとの減少値
        ASC_type: 使わない場合は '', 使う場合は次のいずれか 'SeTranVAE' or 'VQ-SeTranVAE'
        ASC_trajectory_length: 5 # ASC_trajectory_length = 0 is not use, ASC_trajectory_length > 0 
        ASC_mask_probabirity: 0.2 #  集合要素 2 以上の場合の 2 つめ以降の mask 率 TODO: 現在実装されていない
        ASC_dropout: 0.2 # ASC model の dropout 率
    contrastive_learning: 
        use: True # Whether to perform contrastive learning
        temperature: 0.5 # The temperature parameter in contrastive learning 
    loss_coefficient:
        rl: 1.0 # default: 1.0 # RL loss coefficient
        rnd: 1.0 # default: 1.0 # RND loss coefficient
        recon: 1.0 # default: 1.0 # reconstruction loss coefficient for VAE and VQ-VAE
        vae_kl: 1.0 # default: 1.0 # KL loss coefficient for VAE
        codebook: 1.0 # default: 1.0 # reconstruction loss coefficient for VQ-VAE
        commitment: 0.25 # default: 0.25 # commitment coefficient for VQ-VAE
        contrast: 1.0 # default: 1.0  # commitment coefficient for contrastive learning
        recon_p_set: 1.0 # default: 1.0 # policy reconstruction loss coefficient for SeTranVAE and VQ-SeTranVAE
        recon_o_set: 1.0 # default: 1.0 # re_observation reconstruction loss coefficient for SeTranVAE and VQ-SeTranVAE
        cos_weighted: 0.01 # default: 0.1 # re_observation reconstruction cos_weighted loss coefficient for SeTranVAE and VQ-SeTranVAE
        hungarian: 1.0 # default: 0.8 # re_observation reconstruction hungarian loss coefficient for SeTranVAE and VQ-SeTranVAE
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
