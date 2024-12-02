# Usage: python3 scripts/entropy_average_plot.py [任意のlogの数] [任意のタイトル名] [ディレクトリのパス]
# ex) python3 scripts/model_output_get.py 0 RSRS 202402201125
# [任意のlogの数]が0の場合は全てのlogの平均

from re import A
import sys
import numpy as np
import glob
import csv
import pprint
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim

# RSRS
class PVQCModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, 2**hyperplane_n) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        self.head_c = nn.Linear(nn_size, 2**hyperplane_n) # 信頼度(confidence rate)

    def forward(self, x, hidden=None):
        h_l = self.fc1(x)
        h = F.relu(h_l)
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        h_a = self.head_a(h)
        h_b = self.head_b(h)
        h_q = h_b + h_a - h_a.sum(-1).unsqueeze(-1)
        h_c = self.head_c(h)
        return {
            'policy': h_p, 'value': h_v, 
            'advantage_for_q': h_a, 'qvalue': h_q, 'latent': h_l, 'confidence': h_c}

# RSRS(R4D)
class R4DPVQCModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 512
        confidence_size = 32
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n) # policy
        self.head_v = nn.Linear(nn_size, 1) # value
        self.head_a = nn.Linear(nn_size, 2**hyperplane_n) # advantage
        self.head_b = nn.Linear(nn_size, 1) # ベースライン
        # 信頼度
        ## 学習
        self.fc_c = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_c = nn.Linear(nn_size, (2**hyperplane_n)*confidence_size)
        ## 固定
        self.fc_c_fix = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_c_fix = nn.Linear(nn_size, (2**hyperplane_n)*confidence_size)

    def forward(self, x, hidden=None):
        h_l = self.fc1(x)
        h = F.relu(h_l)
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        h_a = self.head_a(h)
        h_b = self.head_b(h)
        h_q = h_b + h_a - h_a.sum(-1).unsqueeze(-1)
        h_c = F.relu(self.fc_c(x))
        h_c = self.head_c(h_c)
        h_c_fix = F.relu(self.fc_c_fix(x))
        h_c_fix = self.head_c_fix(h_c_fix)
        return {
            'policy': h_p, 'value': h_v, 
            'advantage_for_q': h_a, 'qvalue': h_q, 'latent': h_l, 'confidence_57': h_c,
            'confidence_57_fix': h_c_fix}

n = 15

def kernel(n):
    a = np.array(list(range(1, 1 + (n+1)//2)) + list(range(1 + n//2,1,-1)))
    return a / a.sum()


#ここでlogの取得
#pathがlogファイル名
def get_reward_list(path):
    #初期化・定義
    opponents = set()
    #epoch_data_list = [{}]
    epoch_list = [0]
    step_list = [0]
    game_list = [0]

    f = open(path) #fにlogを格納
    lines = f.readlines() #fの中身を全て文字列にする
    prev_line = ''

    #全ての文字列に対してそれぞれの情報を分別して格納する
    for line in lines:
        if line.startswith("{'env_args'"):
            tmp =re.split('[:,]',line)
            hyperplane_n = int(tmp[7])
        #if line.startswith('state'): #startswithで最初の文字列を識別
            #epoch_data_list.append({})
            #epoch_list.append(len(epoch_list))
            #step_list.append(int(line.split('(')[1].rstrip().rstrip(')'))) #rstripで特定の文字を除去
        if line.startswith('state'):
            elms = re.split('[:\n]',line)
            #epoch_data_list.append({})
            #for e in elms[2:]:
            #print(elms[1])
            #name, s = elms.split(':')
            #print(name)
            #print(s)
            s = eval(elms[1])
            epoch_data_list = s
            break
        #if line.startswith('epoch '):
            #if ' ' in prev_line:
                #game = int(prev_line.split()[-1])
                #game_list.append(game)

        #prev_line = line

    #epoch_data_list = [s for s in epoch_data_list if s != {}]
    #game_list = game_list[:len(epoch_data_list)]

    #clipped_epoch_list = epoch_list[n//2:-n//2+1]
    #clipped_step_list = step_list[n//2:-n//2+1]
    #clipped_game_list = game_list[n//2:-n//2+1]
    #null_data = {'r': 0, 'n': 0}
    #kn = kernel(n)
    #averaged_loss_lists = {} #平均報酬のリスト
    #start_epoch = {}
    #for name in epoch_data_list[0].keys():
        #data = [d[name] for d in epoch_data_list]
        #averaged_loss_lists[name] = np.convolve(data, kn, mode='valid')
        #start_epoch[name] = 0
    #return clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_loss_lists, start_epoch
    return epoch_data_list, hyperplane_n


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
    model_log = sorted(glob.glob(path + "latest_***.pth"))

a = int(sys.argv[1]) #使うlogの個数を指定
model_name = sys.argv[2]
if(a == 0 or len(logs) < a):
    a = len(logs)
print(logs[:a]) #平均を出すlog

nn_mean = 0 #平均を仮で入れる
reg_mean = 0
mix_mean = 0
nn_flag = False
reg_flag = False
#policy_mean = []
#v_mean = []
#q_mean = []

for i in logs[:a]:
    # clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_loss_lists, start_epoch = get_reward_list(i) #get_wp_listでlogデータを抽出
    epoch_s_list, hyperplane_n = get_reward_list(i)

    #print(i,averaged_loss_lists)
    #if 'ent_c_nn' in averaged_loss_lists:
        #nn_mean += averaged_loss_lists['ent_c_nn']
        #nn_flag = True
    #if 'ent_c_reg' in averaged_loss_lists:
        #reg_mean += averaged_loss_lists['ent_c_reg']
        #reg_flag = True
    #if 'entropy_c_mixed' in averaged_loss_lists:
        #mix_mean += averaged_loss_lists['entropy_c_mixed']
if model_name == 'RSRS':
    model = PVQCModel(hyperplane_n)
elif model_name == 'R4D-RSRS':
    model = R4DPVQCModel(hyperplane_n)

cnt = 0
for i in model_log[:a]:
    model.load_state_dict(torch.load(i))
    model.eval()
    s_policy = []
    v_value = []
    q_value = []
    reliability = []

    for s in epoch_s_list:
        input = torch.tensor(s)
        output = model(input)
        # 複数の結果を格納して平均する
        s_policy += [output["policy"].tolist()]
        v_value += [output["value"].item()]
        q_value += [output["qvalue"].tolist()]
        if 'confidence_57' in output:
            # エントロピー監視用
            c_predict = output['confidence_57']
            c_target = output['confidence_57_fix']
            c_predict = c_predict.reshape(output['policy'].size(0),-1)
            c_target = c_target.reshape(output['policy'].size(0),-1)
            ## 計算
            bottom = torch.mean((c_target - c_predict)**2, axis =-1) #Σ(真-予測)^2
           
            epsilon = 1e-6
            rnd = epsilon/(bottom+epsilon) #0除算回避
            ## 正規化
            reliability += [(rnd/torch.sum(rnd, keepdim=True, axis=-1)).tolist()]

    if cnt == 0:
        policy_mean = s_policy
        v_mean = v_value
        q_mean = q_value
        reliability_mean = reliability
    elif cnt!=0:
        policy_mean = np.array(policy_mean) + np.array(s_policy)
        v_mean = np.array(v_mean) + np.array(v_value)
        q_mean = np.array(q_mean) + np.array(q_value)
        reliability_mean = np.array(reliability_mean) + np.array(reliability)
    cnt += 1

policy_mean = np.array(policy_mean) / a
v_mean = np.array(v_mean) /a
q_mean = np.array(q_mean) /a
reliability_mean = np.array(reliability_mean) / a



result_list = {"policy": policy_mean, "v": v_mean, "q": q_mean, "reliability": reliability_mean}
#nn_mean =  nn_mean / a #平均を取る
#reg_mean =  reg_mean / a
#mix_mean = mix_mean / a
#if nn_flag == True and reg_flag == True:
#    averaged_loss_lists = {"entropy_c_mixed" : mix_mean, "ent_c_nn" : nn_mean, "ent_c_reg" : reg_mean} #dict型に変換
#elif nn_flag == True and reg_flag == False:
#    averaged_loss_lists = {"ent_c_nn" : nn_mean}
#elif nn_flag == False and reg_flag == True:
#    averaged_loss_lists = {"ent_c_reg" : reg_mean}
#else:
    #print("Not Regional!")
    #sys.exit()
#print("log_average = ",averaged_loss_lists) #指定した個数のファイルの平均勝率を表示

opponents = list(result_list.keys())
#opponents = sorted(opponents_, key=lambda o: result_list[o][-1], reverse=True)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

#last_win_rate = {}
for opponent in opponents:
    if opponent == 'total':
        continue
    l = result_list[opponent]
    #start = start_epoch[opponent]
    #print(f'len(x): {len(clipped_game_list[start:])}')
    #print(f'len(y): {len(loss_list[start:])}')
    #ax.plot(clipped_game_list[start:], loss_list[start:], label=opponent)
    #plt.legend()
    #last_win_rate[opponent] = loss_list[-1]

    if opponent == 'policy':
        np.savetxt(path + 'policy.csv', l, delimiter=',', fmt='%.5f')
        #np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    elif opponent == 'v':
        np.savetxt(path + 'v.csv', l, delimiter=',', fmt='%.5f')
        #np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    elif opponent == 'q':
        np.savetxt(path + 'q.csv', l, delimiter=',', fmt='%.5f')
    elif opponent == 'reliability':
        np.savetxt(path + 'reliability.csv', l, delimiter=',', fmt='%.5f')
        #np.savetxt(path + 'en_episodes.csv', [clipped_game_list[start:]], delimiter=',', fmt='%d')
    #else:
        #np.savetxt(path + 'retruns_' + opponent + '.csv', loss_list[start:], delimiter='', fmt='%.5f')
        #np.savetxt(path + 'episodes_' + opponent + '.csv', clipped_game_list[start:], delimiter='', fmt='%d')

#ax.set_xlabel('episodes', size=14)
#ax.set_ylabel('Average entropy', size=14)

# A corresponding gridss
#plt.grid(which='minor', color='gray', alpha=0.5,
         #linestyle='--', linewidth=0.5)
#plt.grid(which='major', color='gray', alpha=0.5,
         #linestyle='--', linewidth=1)

#fig.tight_layout()
#plt.savefig(path + sys.argv[2] + "_entropy.png")
#plt.show()
