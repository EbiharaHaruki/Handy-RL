#!/bin/bash
# Usage: . bash_scripts/experiment_config.sh [任意の実行回数]

StartDATE=`date +%Y%m%d%H%M` # 実験開始日時を取得
mkdir trainlog/Start$StartDATE # 実験開始日時のディレクトリを作成

N=$1 #標準入力から実験回数を取得

cd configs
file_count=`ls | wc -l` # configsフォルダ内のファイル数を計算
#echo $file_count
cd ..

config_file=config_xxx.yaml # 検索するconfigファイル名(xxxはワイルドカード)

for i in `seq -f %02g 1 $file_count`; do # 指定回数以下を実行
    ###ここでconfigを更新
    config=${config_file//xxx/$i} # configsにconfig_$i.yamlを格納
    rnm_config=`find configs/$config` # configsディレクトリ内から$configを検索してrnm_configに格納
    echo $rnm_config #rnm_configの中身
    mv $rnm_config config.yaml #rnm_configを移動しconfig.yamlにrename

    DATE=`date +%Y%m%d%H%M` #実験日時を取得
    mkdir trainlog/Start$StartDATE/$DATE # StartDATEディレクト下にDateディレクトリを作成
    ex_base="python3 -u main.py --train | tee trainlog/Start$StartDATE/$DATE/train_log_xxx.txt"

    for j in `seq -f %02g 1 $N`; do #指定回数以下を実行
        eval ${ex_base//xxx/$j}
    done

    mv config.yaml trainlog/Start$StartDATE/$DATE/$config
done

#echo @ikedasan $StartDATE の実験やったよ！！ | bash_scripts/slack_alarm.sh #slackに実験が終わったら通知を送る機能（別途で設定する必要あり）

cd trainlog
zip -r Start$StartDATE.zip Start$StartDATE
cd ..