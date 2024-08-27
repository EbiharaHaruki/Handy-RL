# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


def bimap_r(x, y, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(bimap_r(xx, y[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, bimap_r(xx, y[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y) if callback_fn is not None else None


def trimap_r(x, y, z, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(trimap_r(xx, y[i], z[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, trimap_r(xx, y[key], z[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y, z) if callback_fn is not None else None


def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1)
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1))
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1)
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1))
                for key2 in x_front
            )
    return x


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def hungarian(predictions, targets):
    # Compute cosine similarity matrix for the entire batch
    cosine_sim = F.cosine_similarity(predictions.unsqueeze(2), targets.unsqueeze(1), dim=-1) # Shape: (batch_size, predictions_size, targets_size)

    
    # Convert cosine similarity to a cost matrix for the Hungarian algorithm
    cost_matrix = ((1 - cosine_sim)/2).detach().numpy()  # Shape: (batch_size, predictions_size, targets_size)

    # Solve the linear sum assignment problem (Hungarian algorithm)
    index = torch.Tensor([linear_sum_assignment(cost_matrix[i, :, :]) for i in range(predictions.size(0))]).to(torch.long)
    # Store indices
    row_indices = index[:, 0, :].to(predictions.device)
    col_indices = index[:, 1, :].to(predictions.device)
    
    # Gather matched predictions and targets
    matched_predictions = predictions.gather(1, row_indices.unsqueeze(-1).expand(-1, -1, predictions.size(-1)))
    matched_targets = targets.gather(1, col_indices.unsqueeze(-1).expand(-1, -1, predictions.size(-1)))

    return matched_predictions, matched_targets