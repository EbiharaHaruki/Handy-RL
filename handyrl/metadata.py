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
                latent = np.array([md['latent'][p] for md in metadata['metadata']])
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
            # print(f'<><><> self.keys.__sizeof__():{self.keys.__sizeof__()}')
        # print(f'<><><> self.num: {self.num}')
        # print(f'<><><> self.keys shape: {self.keys.shape}')
        # print(f'<><><> lattents shape: {lattents.shape}')
        # print(f'<><><> values shape: {values.shape}')
        # print(f'<><><> lattents: {lattents[0]}')
        # print(f'<><><> values: {values[0]}')

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
            dist, idx = self.nn_index.search(query[np.newaxis, :], self.args['k'])
            # 距離の逆数を計算
            dist_inv = 1.0/(dist+1.0)
            # 近似 vector の計算 
            # print(f'<><><> idx.shape: {idx}')
            # print(f'<><><> self.values[idx[0], :].shape: {self.values[idx[0], :].shape}')
            # print(f'<><><> dist_inv.shape: {dist_inv.shape}')
            return (dist_inv @ self.values[idx[0], :])/dist_inv.sum()
            # regional_confidence = (dist_inv @ self.values[idx[0], :])/dist_inv.sum()
            # print(f'<><><> regional_confidence: {regional_confidence}')
            # return regional_confidence