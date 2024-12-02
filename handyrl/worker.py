# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import random
import threading
import time
import functools
from socket import gethostname
from collections import deque
import multiprocessing as mp
import pickle
import copy
import queue

from .environment import prepare_env, make_env
from .connection import QueueCommunicator
from .connection import send_recv, open_multiprocessing_connections
from .connection import connect_socket_connection, accept_socket_connections
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper, RandomModel


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = -1, None
        self.latest_metadata = -1, None

        self.env = make_env({**args['env'], 'id': wid})
        self.env_e = copy.deepcopy(self.env)
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env_e, self.args)
        self.num_global_episodes = 0

        self.generate_count = 0
        self.eval_count = 0
        self.metadata_id = 0

        self.play_subagent_prob = self.args['agent']['play_subagent_base_prob'] if 'play_subagent_base_prob' in self.args['agent'].keys() else 0.0
        self.play_subagent_lower_prob = self.args['agent']['play_subagent_lower_prob'] if 'play_subagent_lower_prob' in self.args['agent'].keys() else 0.0
        self.play_subagent_decay_per_ep = args['agent']['play_subagent_decay_per_ep'] if 'play_subagent_decay_per_ep' in args['agent'].keys() else 1.0

        random.seed(args['seed'] + wid)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def _gather_models(self, model_ids):
        model_pool = {}
        # 新たに得られた model id のリストに対して roop
        for model_id in model_ids:
            # model pool に入っているかで対応
            if model_id not in model_pool:
                if model_id < 0:
                    # モデル id が 0 未満 = model を持たない player 
                    model_pool[model_id] = None
                elif model_id == self.latest_model[0]:
                    # 初期は self.latest_model[0] = -1 が入っている
                    # 初期は self.latest_model[1] = None が入っている
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model = pickle.loads(send_recv(self.conn, ('model', model_id)))
                    # model 追加を初めてやる時
                    if model_id == 0:
                        # use random model
                        self.env.reset()
                        obs = self.env.observation(self.env.players()[0])
                        # 学習が始まるまではランダム model
                        model = RandomModel(model, {'o':obs})
                    # モデルを格納する
                    model_pool[model_id] = ModelWrapper(model)
                    # update latest model
                    if model_id > self.latest_model[0]:
                        # モデルが存在する時 0 より大きい時に最終モデルを保存しておく
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def _gather_metadatas(self, metadata_ids):
        metadata_pool = {}
        for metadata_id in metadata_ids:
            if metadata_id not in metadata_pool:
                if metadata_id < 0:
                    metadata_pool[metadata_id] = None
                elif metadata_id == self.latest_metadata[0]:
                    # use latest metadata
                    metadata_pool[metadata_id] = self.latest_metadata[1]
                else:
                    # get metadata from server
                    metadata = send_recv(self.conn, ('metadata', metadata_id))
                    metadata = pickle.loads(metadata)
                    metadata_pool[metadata_id] = metadata
                    # update latest metadata
                    if metadata_id > self.latest_metadata[0]:
                        self.latest_metadata = metadata_id, metadata_pool[metadata_id]
        return metadata_pool

    # TODO: 正しい非定常環境として今後再実装
    # def uns_woker(self):
    #     print("worker_uns : ")
    #     self.env.uns()

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            if args is None:
                break
            role = args['role']
            args['play_subagent_prob'] = self.play_subagent_prob          

            models = {}
            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids)

                # make dict of models
                for p, model_id in args['model_id'].items():
                    models[p] = model_pool[model_id]

            metadataset = {}
            if 'metadata_id' in args:
                metadata_ids = list(args['metadata_id'].values())
                metadata_pool = self._gather_metadatas(metadata_ids)
                # make dict of models
                for p, metadata_id in args['metadata_id'].items():
                    metadataset[p] = metadata_pool[metadata_id]
                    self.num_global_episodes = metadataset[p]['num_episodes']
                args['num_global_episodes'] = self.num_global_episodes
            
            if role == 'g':
                episode, return_metadata = self.generator.execute(models, metadataset, args)
                send_recv(self.conn, ('episode', episode))
                send_recv(self.conn, ('return_metadata', return_metadata))
                self.generate_count += 1
                if self.generate_count % self.args['saving_env_status_interval_episodes'] == 0:
                    self.env.fprint_env_status(role, self.worker_id) # 環境の状態ログを出力  
                self.play_subagent_prob = max(self.play_subagent_prob - self.play_subagent_decay_per_ep, self.play_subagent_lower_prob)
            elif role == 'e':
                result, return_metadata = self.evaluator.execute(models, metadataset, args)
                send_recv(self.conn, ('result', result))
                self.eval_count += 1
                if self.eval_count % self.args['saving_env_status_interval_episodes'] == 0:
                    self.env_e.fprint_env_status(role, self.worker_id) # 環境の状態ログを出力 
                # send_recv(self.conn, ('metadata', return_metadata))



def make_worker_args(args, n_ga, gaid, base_wid, wid, conn):
    return args, conn, base_wid + wid * n_ga + gaid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()

class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.metadata = {'id': -1, 'data': None}
        self.data_map = {'model': {}} # , 'metadata': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_parallel'], args['worker']['num_gathers']

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        base_wid = args['worker'].get('base_worker_id', 0)

        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid, base_wid)
        )

        for conn in worker_conns:
            self.add_connection(conn)

        self.buffer_length = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while self.connection_count() > 0:
            try:
                conn, (command, args) = self.recv(timeout=0.3)
            except queue.Empty:
                continue

            if command == 'args':
                # When requested arguments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get multiple arguments from server and store them
                    self.server_conn.send((command, [None] * self.buffer_length))
                    self.args_queue += self.server_conn.recv()
                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command == 'metadata':
                # answer data request as soon as possible
                data_id = args
                if data_id != self.metadata['id']:
                    self.server_conn.send((command, args))
                    self.metadata['id'] = data_id
                    self.metadata['data'] = self.server_conn.recv()
                self.send(conn, self.metadata['data'])

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.buffer_length:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gaid):
    gather = Gather(args, conn, gaid)
    gather.run()


class WorkerCluster(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        # open local connections
        if 'num_gathers' not in self.args['worker']:
            self.args['worker']['num_gathers'] = 1 + max(0, self.args['worker']['num_parallel'] - 1) // 16
        for i in range(self.args['worker']['num_gathers']):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
            conn1.close()
            self.add_connection(conn0)


class WorkerServer(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.total_worker_count = 0

    def run(self):
        # prepare listening connections
        def entry_server(port):
            print('started entry server %d' % port)
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                worker_args = conn.recv()
                print('accepted connection from %s!' % worker_args['address'])
                worker_args['base_worker_id'] = self.total_worker_count
                self.total_worker_count += worker_args['num_parallel']
                args = copy.deepcopy(self.args)
                args['worker'] = worker_args
                conn.send(args)
                conn.close()
            print('finished entry server')

        def worker_server(port):
            print('started worker server %d' % port)
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                self.add_connection(conn)
            print('finished worker server')

        threading.Thread(target=entry_server, args=(9999,), daemon=True).start()
        threading.Thread(target=worker_server, args=(9998,), daemon=True).start()


def entry(worker_args):
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


class RemoteWorkerCluster:
    def __init__(self, args):
        args['address'] = gethostname()
        if 'num_gathers' not in args:
            args['num_gathers'] = 1 + max(0, args['num_parallel'] - 1) // 16

        self.args = args

    def run(self):
        args = entry(self.args)
        print(args)
        prepare_env(args['env'])

        # open worker
        process = []
        try:
            for i in range(self.args['num_gathers']):
                conn = connect_socket_connection(self.args['server_address'], 9998)
                p = mp.Process(target=gather_loop, args=(args, conn, i))
                p.start()
                conn.close()
                process.append(p)
            while True:
                time.sleep(100)
        finally:
            for p in process:
                p.terminate()


def worker_main(args, argv):
    # offline generation worker
    worker_args = args['worker_args']
    if len(argv) >= 1:
        worker_args['num_parallel'] = int(argv[0])

    worker = RemoteWorkerCluster(args=worker_args)
    worker.run()
