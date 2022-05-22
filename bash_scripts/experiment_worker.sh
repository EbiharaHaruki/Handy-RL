#! /bin/bash
# Usage: . bash_scripts/experiment_worker.sh [任意の実行回数]

DATE=`date +%Y%m%d%H%M` #実験日時を取得
mkdir trainlog/$DATE #実験日時のディレクトリ作成
time=30  #timeoutまでの時間

N=$1 #標準入力から実験回数を取得
ex_base="timeout $time python3 -u main.py --worker" #学習
#current_conda=$CONDA_DEFAULT_ENV

for i in `seq -f %02g 1 $N`; do #指定回数以下を実行
    eval ${ex_base//xxx/$i}
    sleep 5 #サーバーとの兼ね合いで少し時間置いてから実行
done
