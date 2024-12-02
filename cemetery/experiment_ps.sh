#! /bin/bash

ProcessName=experiment_ps.sh #プロセス名指定
PID=$$ #実行している自身のプロセスIDを取得

#ここで一定時間経過したプロセスをkillする
#for文が回ってない？
for i in `ps -ef | grep "$ProcessName" || grep -v $PID || grep -v grep | awk '{print $2}'`
do
  echo "aaa"
  TIME = $(`ps -o lstart -noheader -p $i`)  #プロセスIDの起動時間の取得

  #このif文はTIMEが取得できなかった時に経過時間を１秒とする処理
  if [ -n "$TIME" ]; then
    StartupTime=`date +%s -d "$TIME"` #起動時刻をUNIX時刻に変換
    CurrentTime=`date +%s` #現在の時刻の変換
    ElapsedTime=`expr $CurrentTime - $StartupTime` #経過時間の取得
  else
    ElapsedTime=1
  fi

  #指定した秒数以上経過でプロセスIDをKill
  if [ $ElapsedTime -gt 180 ] ; then
    kill $i
  fi

done
