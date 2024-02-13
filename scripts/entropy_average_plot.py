# Usage: python3 scripts/entropy_average_plot.py [任意のlogの数] [任意のタイトル名] [ディレクトリのパス]
# ex) python3 scripts/entropy_average_plot.py 0 entropy_result 202301010101
# [任意のlogの数]が0の場合は全てのlogの平均

from re import A
import sys
import numpy as np
import glob
import csv
import pprint

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
        if line.startswith('loss'):
            elms = line.split()
            epoch_data_list.append({})
            for e in elms[2:]:
                name, loss = e.split(':')
                if name == "ent_c_nn" or name=="ent_c_reg" or name=="ent_c_mixed":
                    loss = float(loss)
                    epoch_data_list[-1][name] = loss
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
    averaged_loss_lists = {} #平均報酬のリスト
    start_epoch = {}
    for name in epoch_data_list[0].keys():
        data = [d[name] for d in epoch_data_list]
        averaged_loss_lists[name] = np.convolve(data, kn, mode='valid')
        start_epoch[name] = 0
    return clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_loss_lists, start_epoch


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

a = int(sys.argv[1]) #使うlogの個数を指定
if(a == 0 or len(logs) < a):
    a = len(logs)
print(logs[:a]) #平均を出すlog
nn_mean = 0 #平均を仮で入れる
reg_mean = 0
mix_mean = 0
nn_flag = False
reg_flag = False
for i in logs[:a]:
    clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_loss_lists, start_epoch = get_reward_list(i) #get_wp_listでlogデータを抽出
    print(i,averaged_loss_lists)
    if 'ent_c_nn' in averaged_loss_lists:
        nn_mean += averaged_loss_lists['ent_c_nn']
        nn_flag = True
    if 'ent_c_reg' in averaged_loss_lists:
        reg_mean += averaged_loss_lists['ent_c_reg']
        reg_flag = True
    if 'ent_c_mixed' in averaged_loss_lists:
        mix_mean += averaged_loss_lists['ent_c_mixed']

nn_mean =  nn_mean / a #平均を取る
reg_mean =  reg_mean / a
mix_mean = mix_mean / a
if nn_flag == True and reg_flag == True:
    averaged_loss_lists = {"ent_c_mixed" : mix_mean, "ent_c_nn" : nn_mean, "ent_c_reg" : reg_mean} #dict型に変換
elif nn_flag == True and reg_flag == False:
    averaged_loss_lists = {"ent_c_mixed" : mix_mean, "ent_c_nn" : nn_mean}
elif nn_flag == False and reg_flag == True:
    averaged_loss_lists = {"ent_c_mixed" : mix_mean, "ent_c_reg" : reg_mean}
else:
    print("Not Regional!")
    sys.exit()
print("log_average = ",averaged_loss_lists) #指定した個数のファイルの平均勝率を表示

opponents_ = list(averaged_loss_lists.keys())
opponents = sorted(opponents_, key=lambda o: averaged_loss_lists[o][-1], reverse=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

last_win_rate = {}
for opponent in opponents:
    if opponent == 'total':
        continue
    loss_list = averaged_loss_lists[opponent]
    start = start_epoch[opponent]
    print(f'len(x): {len(clipped_game_list[start:])}')
    print(f'len(y): {len(loss_list[start:])}')
    ax.plot(clipped_game_list[start:], loss_list[start:], label=opponent)
    plt.legend()
    last_win_rate[opponent] = loss_list[-1]

    if opponent == 'ent_c_nn':
        np.savetxt(path + 'entropy_nn.csv', [loss_list[start:]], delimiter=',', fmt='%.5f')
        np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    elif opponent == 'ent_c_reg':
        np.savetxt(path + 'entropy_reg.csv', [loss_list[start:]], delimiter=',', fmt='%.5f')
        np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    elif opponent == 'ent_c_mixed':
        np.savetxt(path + 'entropy_mix.csv', [loss_list[start:]], delimiter=',', fmt='%.5f')
        np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    else:
        np.savetxt(path + 'retruns_' + opponent + '.csv', loss_list[start:], delimiter='', fmt='%.5f')
        np.savetxt(path + 'episodes_' + opponent + '.csv', clipped_game_list[start:], delimiter='', fmt='%d')

ax.set_xlabel('episodes', size=14)
ax.set_ylabel('Average entropy', size=14)

# A corresponding gridss
plt.grid(which='minor', color='gray', alpha=0.5,
         linestyle='--', linewidth=0.5)
plt.grid(which='major', color='gray', alpha=0.5,
         linestyle='--', linewidth=1)

fig.tight_layout()
plt.savefig(path + sys.argv[2] + "_entropy.png")
plt.show()
