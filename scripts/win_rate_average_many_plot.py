# Usage: python3 scripts/win_rate_average_many_plot.py [任意のlogの数] [任意のタイトル名] [ディレクトリのパス]
# [任意のlogの数]が0の場合は全てのlogの平均

from re import A
import sys
from matplotlib import rc_params_from_file
import numpy as np
import glob
import os
import re

#この数字はなんだ
n = 15

def kernel(n):
    a = np.array(list(range(1, 1 + (n+1)//2)) + list(range(1 + n//2,1,-1)))
    return a / a.sum()


#ここでlogの取得
#pathがlogファイル名
def get_wp_list(path):
    #初期化・定義
    opponents = set()
    config_list = []
    epoch_data_list = [{}]
    epoch_list = [0]
    step_list = [0]
    game_list = [0]

    f = open(path) #fにlogを格納
    lines = f.readlines() #fの中身を全て文字列にする
    prev_line = ''

    #全ての文字列に対してそれぞれの情報を分別して格納する
    for line in lines:
        if line.startswith("{'env_args':"): #configのパラメータ情報を取得
            prm = re.findall("'entropy_regularization': .....", line) #エントロピーの文字列を取得（他の文字列に置き換えることも可能）
            depth = re.findall("'depth': ...", line)
            coordinate = re.findall("'hyperplane_n': ..", line)
            config_list.append(prm)
            config_list.append(depth)
            config_list.append(coordinate)
        if line.startswith('updated'): #startswithで最初の文字列を識別
            epoch_data_list.append({})
            epoch_list.append(len(epoch_list))
            step_list.append(int(line.split('(')[1].rstrip().rstrip(')'))) #rstripで特定の文字を除去
        if line.startswith('win rate'):
            elms = line.split()
            opponent = elms[2].lstrip('(').rstrip(')')
            games = int(elms[-1].lstrip('(').rstrip(')'))
            wp = float(elms[-4]) if games > 0 else 0.0
            epoch_data_list[-1][opponent] = {'w': games * wp, 'n': games}
            opponents.add(opponent)
        if line.startswith('epoch '):
            #print(line, len(epoch_list))
            if ' ' in prev_line:
                game = int(prev_line.split()[-1])
                game_list.append(game)

        prev_line = line

    game_list = game_list[:len(epoch_data_list)]

    clipped_epoch_list = epoch_list[n//2:-n//2+1]
    clipped_step_list = step_list[n//2:-n//2+1]
    clipped_game_list = game_list[n//2:-n//2+1]
    null_data = {'w': 0, 'n': 0}
    kn = kernel(n)
    #print(kn)
    averaged_wp_lists = {} #平均勝率のリスト
    start_epoch = {}
    for opponent in opponents:
        win_list = [e.get(opponent, null_data)['w'] for e in epoch_data_list]
        n_list = [e.get(opponent, null_data)['n'] for e in epoch_data_list]
        averaged_win_list = np.convolve(win_list, kn, mode='valid')
        averaged_n_list = np.convolve(n_list, kn, mode='valid') + 1e-6
        averaged_wp_lists[opponent] = averaged_win_list / averaged_n_list
        try:
            start_epoch[opponent] = next(i for i, n in enumerate(n_list) if n >= 1)
        except:
            start_epoch[opponent] = 0
    return config_list, clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_wp_lists, start_epoch


import matplotlib.pyplot as plt
import seaborn as sns
flatui = ["#9b59b6", "#95a5a6", "#34495e", "#3498db", "#e74c3c", "#2ecc71", "#b22222"]
sns.set_palette(sns.color_palette(flatui, 24))

files = os.listdir(sys.argv[3]) #対象のディレクトリ名を標準入力の３つ目から受け取る
files_dir = [f for f in files if os.path.isdir(os.path.join(sys.argv[3], f))] #受け取ってるディレクトリの中のログディレクト名を取得
print(files_dir) #上の詳細表示
files_dir.sort()
# print(len(files_dir))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for fname in files_dir: #ディレクトリの中のログディレクトの個数for文を回す
    print(fname)
    if len(sys.argv) != 4: #標準入力が4つなければ実行コマンドが違うのでエラー処理
        print("error : 実行方法が正しくないかも")
        exit()
    else:
        path_name = sys.argv[3]+ "/" + fname + "/" #正しい実行であればpathにログディレクトリまでのパスを入れる
        print(path_name)
        logs = sorted(glob.glob(path_name + "train_log_***.txt")) #ログファイルの検索
    #print(logs)

    a = int(sys.argv[1]) #使うlogの個数を標準入力から指定
    if(a == 0 or len(logs) < a): #もし指定した数が0かログファイルの個数より少ない場合は全部のログからプロット
        a = len(logs)
    #print(a)
    print(logs[:a]) #平均を出すlog
    mean = 0 #平均を仮で入れる
    for j in logs[:a]:
        config_list, clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_wp_lists, start_epoch = get_wp_list(j) #get_wp_listでlogデータを抽出
        print(j,averaged_wp_lists)
        mean += averaged_wp_lists['='] #logから得た勝率を足す
    mean =  mean / a #平均を取る
    averaged_wp_lists = {"=" : mean} #dict型に変換
    print("log_average = ",averaged_wp_lists) #指定した個数のファイルの平均勝率を表示

    opponents_ = list(averaged_wp_lists.keys())
    opponents = sorted(opponents_, key=lambda o: averaged_wp_lists[o][-1], reverse=True)

    last_win_rate = {}
    for opponent in opponents:
        if opponent == 'total':
            continue
        wp_list = averaged_wp_lists[opponent]
        start = start_epoch[opponent]
        # ax.plot(clipped_epoch_list[start:], wp_list[start:], label=opponent)
        label_name = str(config_list[0]) #ラベルの名称取得
        label_name = label_name.replace(',','').replace("'", '').replace("[", '').replace("]", '').replace('"', '').rstrip("e").replace("entropy_regularization", '').replace(':', '') #ラベルの名称から邪魔な文字を消す
        label_name = "entropy regularization: " + str(label_name)
        ax.plot(clipped_game_list[start:], wp_list[start:], label=label_name)
        last_win_rate[opponent] = wp_list[-1]

depth_num = str(config_list[1])
coordinate_num = str(config_list[2])

depth_num = int(depth_num.replace(',','').replace("'", '').replace("[", '').replace("]", '').replace('"', '').replace("depth", '').replace(':', ''))
coordinate_num = int(coordinate_num.replace(',','').replace("'", '').replace("[", '').replace("]", '').replace('"', '').replace("hyperplane_n", '').replace(':', ''))

goal_state_num = (depth_num + 1)**coordinate_num
return_p= float(1 / goal_state_num) #報酬をもらえる確率（報酬エリアが１つの場合）
rlabel= "random return level: "+str('{:.5f}'.format(return_p))

plt.hlines(return_p, clipped_game_list[0], clipped_game_list[-1], 'r', linestyles=':', lw=1, label=rlabel )

ax.set_xlabel('Games', size=14)
ax.set_ylabel('Average win rate', size=14)
ax.set_title(sys.argv[2])
ax.set_ylim(0, 1)
ax.legend()

# Major ticks every 20, minor ticks every 5
major_ticks = np.linspace(0, 1, 11)
minor_ticks = np.linspace(0, 1, 21)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# A corresponding gridss
plt.grid(which='minor', color='gray', alpha=0.5,
         linestyle='--', linewidth=0.5)
plt.grid(which='major', color='gray', alpha=0.5,
         linestyle='--', linewidth=1)

fig.tight_layout()
plt.savefig(sys.argv[3]+ "/" + sys.argv[2] + ".png")
plt.show()