# Usage: . bash_scripts/experiment_nwindow.sh [任意の実行回数]
# tmuxでウィンドウを生成しないバージョン（エラーが確認できない）
# tmux上で実行する必要あり
# timeoutを使っているのは ctrl+c 対策
#! /bin/bash

DATE=`date +%Y%m%d%H%M`
mkdir trainlog/$DATE #実験日時のディレクトリ作成

N=$1 #標準入力から実験回数を取得
time=270  #timeoutまでの時間
ex_base="python3 -u main.py --train | tee ./trainlog/$DATE/train_log_xx.txt" #ログを取りながら学習させる
current_conda=$CONDA_DEFAULT_ENV
num_i=01
tmux new-window -n 'w'$num_i

for j in `seq -f %02g 1 $N`; do #指定回数以下を実行
    #num_i=$(printf "%02d" `expr $j`) #num_iに回数を格納
    #tmux new-window -n 'w'$num_i #num_iという名前を付けたウィンドウを作成
    tmux send-keys -t $num_i "conda activate $current_conda" C-m
    ex_execution=${ex_base//xx/$j} #ex_baseのxx部分に現在のを追加
    #tmux send-keys -t $num_i "$ex_execution" C-m
    tmux send-keys -t $num_i "timeout $time $ex_execution" C-m #ウィンドウ指定でtimeoutを設定して学習させる
    sleep `expr $time` #確実に終了させるために一定時間スリープ
    echo $ex_execution
done

#tmux send-keys -t $num_i "grep finished -rl trainlog/$DATE | sort | tee trainlog/$DATE/finish_log_list.txt" C-m #正常に学習が終了したファイルを検索するやつ
tmux send-keys -t $num_i "echo @ikedasan $DATE の実験終わったよ！！ | bash_scripts/slack_alarm.sh" C-m #slackに実験が終わったら通知を送る機能（別途で設定する必要あり）

zip -r trainlog/$DATE/$DATE.zip trainlog/$DATE #zipファイル生成

ex_plot="python3 scripts/win_rate_average_plot.py 0 sample $DATE" #回数分のログの可視化
tmux send-keys -t $num_i "timeout 10 $ex_plot" C-m
sleep `expr 20`

#課題点
##スクリプトの時間制限をなくして連続実行できるようにする（高難易度）

#問題点
##logファイルのepoch数が違うと平均をプロットできない