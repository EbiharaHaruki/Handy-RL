# 計算機で実行する場合のDockerfile
計算機を使用して実験を回す場合のDockerfile、GPUの使用は想定していない。

## 起動方法
このREADME.mdに移動して下記を実行する。

```
# build
$ cd calculator
$ uid=$(id -u `whoami`)
$ docker build -t handyrl-simpletask-cpu --build-arg useruid=$uid -f calculator/Dockerfile .

# run
$ docker run -it handyrl-simpletask-cpu:latest /bin/bash
```
