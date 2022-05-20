#/bin/bash
ext="train_log_**"

move_file(){
    sendDir=$1
    receiveDir=$2

    if [ ! -d $sendDir ]; then
        echo "Error : Not found [$sendDir]"
        return
    fi

    if [ ! -d $receiveDir ]; then
        echo "Error : Not found [$receiveDir]"
        return
    fi

    cd $sendDir

    for fName in `find . -name "$ext" -and -not `
    do
        efName=`echo $fName`
        receiveFile=$receiveDir/$efName

        if [ -f $receiveFile ]; then
            echo "Error : [$receiveDir] already exists"
            return
        fi

        echo "[$fName] →→ [$receiveDir]"
        mv $fName $receiveFile

        if [ ! -f $receiveFile ]; then
            echo "Error : [$receiveFile] Not found"
            return
        fi
    done
}

DATE=`date +%Y%m%d%H%M`
pass=`pwd`
#echo $DATE

#tmux send-keys "echo $DATE" C-m

#tmux send-keys "cd train_log" C-m
#tmux send-keys "mkdir $DATE" C-m
#tmux send-keys "cd .." C-m

cd trainlog
mkdir $DATE
move_file $pass $pass/trainlog/$DATE
cd $pass