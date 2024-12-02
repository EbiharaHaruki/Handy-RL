# metadata classes

import os
# TODO: OpenMP runtime error をちゃんと解決する（現在は暫定対応） 
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# import time
# import random
import faiss

import numpy as np


def feed_knn(knn, args, return_metadata):
    # store metadata
    # return_metadata[並列蓄積 feadbuck 数][dict][step 数][dict][player]
    for metadata in return_metadata:
        # metadata[dict][step 数][dict][player]
        if metadata is None:
            continue
        if 'knn' in args['metadata']['name']:
            for p in metadata['args']['player']:
                latent = np.array([md['rl_latent'][p] for md in metadata['metadata']])
                actions = np.array([md['action'][p] for md in metadata['metadata']])
                knn.feed(latent, actions)
            knn.update_nn_index()
    return None

class KNN:
    # episodic controll や RS 系 agent が持つ用に
    def __init__(self, args, net=None, remote=False):
        self.args = args['metadata']['knn']
        self.nn_index = None
        self.lastidx = 0
        self.num = 0

        self.zeta = 0.008
        self.epsilon = 0.0001
    
    def _init_memory(self, lattents, values):
        _keys = np.zeros((self.args['size'], lattents.shape[1]), dtype=float)
        _values = np.zeros((self.args['size'], values.shape[1]), dtype=int)
        return _keys, _values

    def feed(self, lattents, values):
        l = lattents.shape[0] # 新規格納数
        li = self.lastidx + l
        # 初期値
        if self.num == 0:
            self.keys, self.values = self._init_memory(lattents, values)
        # 周回保存
        if li == self.args['size']:
            li = l
            self.keys[0:li] = lattents
            self.values[0:li] = values           
        elif li > self.args['size']:
            li = li - self.args['size']
            d = l - li
            self.keys[self.lastidx:self.args['size']] = lattents[0:d]
            self.values[self.lastidx:self.args['size']] = values[0:d]           
            self.keys[0:li] = lattents[d:]
            self.values[0:li] = values[d:]           
        else:
            self.keys[self.lastidx:li] = lattents
            self.values[self.lastidx:li] = values
        # max num update
        if self.num == self.args['size']:
            pass
        elif self.num + l > self.args['size']:
            self.num = self.args['size']
        else:
            self.num += l # 格納数を更新
        self.lastidx = li

    def update_nn_index(self):
        _keys = self.keys[0:self.num, :]
        # _values = self.values[0:self.num, :]
        self.nn_index = faiss.IndexFlatL2(_keys.shape[1])
        self.nn_index.add(_keys)

    def regional_nn(self, query):
        # search 
        if (self.nn_index is None) or (self.num < self.args['k']):
        # if (self.nn_index is None):
            return None
        else:
            if len(query.shape) == 1:
                query = query[np.newaxis, :]
            dist, idx = self.nn_index.search(query, self.args['k'])
            # 距離の逆数を計算
            # dist_inv = 1.0/(dist+1.0)

            # 平方ユークリッド距離(ユークリッドの2乗)を計算しd_kに格納
            d_k = dist # .squeeze() # ** 2
            # d_kを使ってユークリッド距離の移動平均d^2_mを計算
            d_m = d_k.mean(axis = 0)
            # カーネル値の分母の分数を計算(d_kの正則化)
            # d_n = d_k / d_m_ave # 0 除算の代わりに 0 を置き換える
            d_n = np.divide(d_k, d_m, out=np.zeros_like(d_k), where=d_m != 0.0) - self.zeta
            # d_n があまりに小さい場合 0 に更新
            d_n[d_n < 0.0] = 0.0
            # 入力と近傍値のカーネル値（類似度）k_v を計算
            k_v = self.epsilon / (d_n + self.epsilon)
            # 類似度K_vから総和が1となる重み生成。疑似試行回数 n の総和を1にしたいため
            weight = (k_v / k_v.sum(axis = 1)).squeeze()
            # weight = k_v.squeeze()
            # 類似度から算出した重みと action vector で加重平均を行い疑似試行割合を計算
            # regional_confidence = np.empty((idx.shape[0], self.values.shape[1]))
            # [self.store(regional_confidence, j, np.average(self.values[idx[j,:], :], weights=weight[j,:], axis=0)) for j in range(idx.shape[0])]
            regional_confidence = np.average(self.values[idx.squeeze(), :], weights=weight, axis=0)

            # 近似 vector の計算 
            # return (dist_inv @ self.values[idx[0], :])/dist_inv.sum()
            # regional_confidence = (dist_inv @ self.values[idx[0], :])/dist_inv.sum()
            return regional_confidence

    def store(self, npa, j, v):
        npa[j] = v