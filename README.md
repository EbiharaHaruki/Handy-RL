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

<!-- ## ドキュメント

* [**Config Parameters**](docs/parameters.md) shows a list of parameters of `config.yaml`.
* [**Large Scale Training**](docs/large_scale_training.md) is a procedure for large scale training remotely.
* [**Train with Customized Environment**](docs/custom_environment.md) explains an interface of environment to create your own game.
* [**API**](docs/api.md) shows entry-point APIs of `main.py` -->
