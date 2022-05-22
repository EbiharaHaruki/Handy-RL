#! /bin/bash
# Usage: . bash_scripts/experiment_learner.sh [任意の実行回数]

DATE=`date +%Y%m%d%H%M` #実験日時を取得
mkdir trainlog/$DATE #実験日時のディレクトリ作成

N=$1 #標準入力から実験回数を取得
ex_base="python3 -u main.py --train-server | tee trainlog/$DATE/train_log_xxx.txt" #ログを取りながら学習させる
#current_conda=$CONDA_DEFAULT_ENV

for i in `seq -f %02g 1 $N`; do #指定回数以下を実行
    eval ${ex_base//xxx/$i}
done

echo @ikedasan $DATE の実験 $N 回やったよ！！ | bash_scripts/slack_alarm.sh #slackに実験が終わったら通知を送る機能（別途で設定する必要あり）

plot_now="timeout 10 python3 scripts/win_rate_average_plot.py 0 sample $DATE" #回数分のログの可視化
eval $plot_now
cd trainlog
zip -r trainlog/$DATE/$DATE.zip $DATE
cd ..