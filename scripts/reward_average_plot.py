# Usage: python3 scripts/reward_average_plot.py [任意のlogの数] [任意のタイトル名] [ディレクトリのパス]
# [任意のlogの数]が0の場合は全てのlogの平均

from re import A
import sys
import numpy as np
import glob

n = 15

def kernel(n):
    a = np.array(list(range(1, 1 + (n+1)//2)) + list(range(1 + n//2,1,-1)))
    return a / a.sum()


#ここでlogの取得
#pathがlogファイル名
def get_reward_list(path):
    #初期化・定義
    opponents = set()
    epoch_data_list = [{}]
    epoch_list = [0]
    step_list = [0]
    game_list = [0]

    f = open(path) #fにlogを格納
    lines = f.readlines() #fの中身を全て文字列にする
    prev_line = ''

    #全ての文字列に対してそれぞれの情報を分別して格納する
    for line in lines:
        if line.startswith('updated'): #startswithで最初の文字列を識別
            epoch_data_list.append({})
            epoch_list.append(len(epoch_list))
            step_list.append(int(line.split('(')[1].rstrip().rstrip(')'))) #rstripで特定の文字を除去
        if line.startswith('average reward'):
            elms = line.split()
            games = int(elms[-1].lstrip('(').rstrip(')'))
            reward = float(elms[-4])
            opponent = elms[2].lstrip('(').rstrip(')')
            epoch_data_list[-1][opponent] = {'r': reward, 'n': games}
            opponents.add(opponent)
        if line.startswith('epoch '):
            if ' ' in prev_line:
                game = int(prev_line.split()[-1])
                game_list.append(game)

        prev_line = line

    epoch_data_list = [s for s in epoch_data_list if s != {}]
    game_list = game_list[:len(epoch_data_list)]

    clipped_epoch_list = epoch_list[n//2:-n//2+1]
    clipped_step_list = step_list[n//2:-n//2+1]
    clipped_game_list = game_list[n//2:-n//2+1]
    null_data = {'r': 0, 'n': 0}
    kn = kernel(n)
    averaged_reward_lists = {} #平均報酬のリスト
    start_epoch = {}
    for opponent in opponents:
        reward_list = [e.get(opponent, null_data)['r'] for e in epoch_data_list]
        n_list = [e.get(opponent, null_data)['n'] for e in epoch_data_list]
        averaged_reward_list = np.convolve(reward_list, kn, mode='valid')
        averaged_reward_lists[opponent] = averaged_reward_list
        try:
            start_epoch[opponent] = next(i for i, n in enumerate(n_list) if n >= 1)
        except:
            start_epoch[opponent] = 0
    return clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_reward_lists, start_epoch


import matplotlib.pyplot as plt
import seaborn as sns
flatui = ["#9b59b6", "#95a5a6", "#34495e", "#3498db", "#e74c3c", "#2ecc71", "#b22222"]
sns.set_palette(sns.color_palette(flatui, 24))

if len(sys.argv) != 4:
    path = ""
    logs = sorted(glob.glob("train_log_***.txt"))
else:
    path = "./trainlog/" + sys.argv[3] + "/"
    print(path)
    logs = sorted(glob.glob(path + "train_log_***.txt")) #logファイルの検索
#print(type(sys.argv[3]))
#print(logs)

a = int(sys.argv[1]) #使うlogの個数を指定
if(a == 0 or len(logs) < a):
    a = len(logs)
#print(a)
print(logs[:a]) #平均を出すlog
mean = 0 #平均を仮で入れる
for i in logs[:a]:
    clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_reward_lists, start_epoch = get_reward_list(i) #get_wp_listでlogデータを抽出
    print(i,averaged_reward_lists)
    mean += averaged_reward_lists['='] #logから得た勝率を足す
mean =  mean / a #平均を取る
averaged_reward_lists = {"=" : mean} #dict型に変換
print("log_average = ",averaged_reward_lists) #指定した個数のファイルの平均勝率を表示


opponents_ = list(averaged_reward_lists.keys())
opponents = sorted(opponents_, key=lambda o: averaged_reward_lists[o][-1], reverse=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

last_win_rate = {}
for opponent in opponents:
    if opponent == 'total':
        continue
    reward_list = averaged_reward_lists[opponent]
    start = start_epoch[opponent]
    # ax.plot(clipped_epoch_list[start:], wp_list[start:], label=opponent)
    ax.plot(clipped_game_list[start:], reward_list[start:], label=opponent)
    last_win_rate[opponent] = reward_list[-1]

ax.set_xlabel('episodes', size=14)
ax.set_ylabel('Average reward', size=14)
ax.set_title(sys.argv[2])
# ax.set_ylim(0, 1)
# ax.legend()

# Major ticks every 20, minor ticks every 5
# major_ticks = np.linspace(0, 1, 11)
# minor_ticks = np.linspace(0, 1, 21)

# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

# A corresponding gridss
plt.grid(which='minor', color='gray', alpha=0.5,
         linestyle='--', linewidth=0.5)
plt.grid(which='major', color='gray', alpha=0.5,
         linestyle='--', linewidth=1)

fig.tight_layout()
plt.savefig(path + sys.argv[2] + ".png")
plt.show()