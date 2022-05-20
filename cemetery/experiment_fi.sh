#! /usr/bin/env bash


N=${1}


ex_base="python3 main.py --train | tee train_log_xx.txt"


for i in `seq -f %02g 1 $N`; do #＼＼
  num_i=$(printf "%02d" `expr $i`)
  tmux new-window -n 'w'$num_i
  tmux send-keys -t $num_i 'cd ~/SimpleTask' C-m
  ex_execution=${ex_base//xx/$i}
  tmux send-keys -t $num_i "$ex_execution" C-m
  # echo $ex_execution
done
