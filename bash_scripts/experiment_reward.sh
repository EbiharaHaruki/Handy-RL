#! /bin/bash
# Usage: . bash_scripts/experiment_reward.sh [任意の実行回数] [environment の ファイル名, .py 除く]

DATE=`date +%Y%m%d%H%M` #実験日時を取得
mkdir trainlog/$DATE #実験日時のディレクトリ作成

N=$1 #標準入力から実験回数を取得
ENV=$2
# 最初の文字を取得(gym)
FIRST_STR=${ENV:0:3}
SUBSTRING=${ENV:4}

eval "cp config.yaml trainlog/$DATE/config.yaml"
# 分岐処理
if [ "$FIRST_CHAR" = "gym" ]; then
    eval "cp handyrl/envs/$ENV.py trainlog/$DATE/$SUBSTRING.py"
else
    eval "cp handyrl/envs/$ENV.py trainlog/$DATE/$ENV.py"
fi

ex_base="python3 -u main.py --train | tee trainlog/$DATE/train_log_xxx.txt" #ログを取りながら学習させる
model_cp="cp models/latest.pth trainlog/$DATE/latest_xxx.pth" #最終モデルも保存
#current_conda=$CONDA_DEFAULT_ENV

for i in `seq -f %02g 1 $N`; do #指定回数以下を実行
    #num_i=$(printf "%02d" `expr $i`) #num_iに回数を格納
    #tmux send-keys -t $num_i "conda activate $current_conda" C-m
    #ex_execution=${ex_base//xxx/$i} #ex_baseのxx部分に現在のを追加しコマンド実行
    #eval ${ex_execution}
    eval ${ex_base//xxx/$i}
    eval ${model_cp//xxx/$i}
done

# echo @kumejun $DATE の実験 $N 回やったよ！！ | bash_scripts/slack_alarm.sh #slackに実験が終わったら通知を送る機能（別途で設定する必要あり）

plot_now="timeout 10 python3 scripts/reward_average_plot.py $N $SUBSTRING $DATE" #回数分のログの可視化
eval $plot_now
cd trainlog
zip -r $DATE.zip $DATE
mv $DATE.zip $DATE/
cd ..