
# Usage: python3 scripts/win_rate_plot.py train-log-002.txt all

# You should not include figures generated by this script in your academic paper, because
# 1. This version of HandyRL doesn't display all the results of the matches.
# 2. Smoothing method in this script is not a simple moving average.

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


n = 15


def kernel(n):
    a = np.array(list(range(1, 1 + (n+1)//2)) + list(range(1 + n//2,1,-1)))
    return a / a.sum()


def get_wp_list(path):
    opponents = set()
    epoch_data_list = [{}]
    epoch_list = [0]
    step_list = [0]
    game_list = [0]

    f = open(path)
    lines = f.readlines()
    prev_line = ''

    for line in lines:
        if line.startswith('updated'):
            epoch_data_list.append({})
            epoch_list.append(len(epoch_list))
            step_list.append(int(line.split('(')[1].rstrip().rstrip(')')))
        if line.startswith('win rate'):
            elms = line.split()
            opponent = elms[2].lstrip('(').rstrip(')') if elms[2] != '=' else 'total'
            games = int(elms[-1].lstrip('(').rstrip(')'))
            wp = float(elms[-4]) if games > 0 else 0.0
            epoch_data_list[-1][opponent] = {'w': games * wp, 'n': games}
            opponents.add(opponent)
        if line.startswith('epoch '):
            print(line, len(epoch_list))
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
    averaged_wp_lists = {}
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
    return clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_wp_lists, start_epoch


flatui = ["#9b59b6", "#95a5a6", "#34495e", "#3498db", "#e74c3c", "#2ecc71", "#b22222"]
sns.set_palette(sns.color_palette(flatui, 24))

clipped_epoch_list, clipped_step_list, clipped_game_list, averaged_wp_lists, start_epoch = get_wp_list(sys.argv[1])

opponents_ = list(averaged_wp_lists.keys())
opponents = sorted(opponents_, key=lambda o: averaged_wp_lists[o][-1], reverse=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

last_win_rate = {}
for opponent in opponents:
    wp_list = averaged_wp_lists[opponent]
    start = start_epoch[opponent]
    # ax.plot(clipped_epoch_list[start:], wp_list[start:], label=opponent)
    ax.plot(clipped_game_list[start:], wp_list[start:], label=opponent)
    last_win_rate[opponent] = wp_list[-1]

ax.set_xlabel('Games', size=14)
ax.set_ylabel('Win rate', size=14)
ax.set_title(sys.argv[2])
ax.set_ylim(0, 1)
ax.legend()

# Major ticks every 20, minor ticks every 5
major_ticks = np.linspace(0, 1, 11)
minor_ticks = np.linspace(0, 1, 21)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# A corresponding grid
plt.grid(which='minor', color='gray', alpha=0.5,
         linestyle='--', linewidth=0.5)
plt.grid(which='major', color='gray', alpha=0.5,
         linestyle='--', linewidth=1)

fig.tight_layout()
plt.show()
