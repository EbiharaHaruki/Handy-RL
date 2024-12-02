#!/bin/bash
# Usage: . bash_scripts/experiment_config_worker.sh [任意の実行回数]

time=130  #timeoutまでの時間
N=$1 #標準入力から実験回数を取得

cd configs
file_count=`ls | wc -l` # configsフォルダ内のファイル数を計算
#echo $file_count
cd ..

#finishtime=$time*$file_count*$N
#echo @ikedasan $StartDATE の実験始めるよ！！ | bash_scripts/slack_alarm.sh #slackに実験が終わったら通知を送る機能（別途で設定する必要あり）

config_file=config_xxx.yaml # 検索するconfigファイル名(xxxはワイルドカード)

for i in `seq -f %02g 1 $file_count`; do # 指定回数以下を実行
    ###ここでconfigを更新
    config=${config_file//xxx/$i} # configsにconfig_$i.yamlを格納
    rnm_config=`find configs/$config` # configsディレクトリ内から$configを検索してrnm_configに格納
    echo $rnm_config #rnm_configの中身
    mv $rnm_config config.yaml #rnm_configを移動しconfig.yamlにrename

    ex_base="timeout $time python3 -u main.py --worker" #学習

    for j in `seq -f %02g 1 $N`; do #指定回数以下を実行
        eval ${ex_base//xxx/$j}
        sleep 3 #サーバーとの兼ね合いで少し時間置いてから実行
    done

done