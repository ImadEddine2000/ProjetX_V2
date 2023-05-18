import numpy as np
import os
import pandas as pd

SIM_PATH = os.path.join("", "result", "sim.npy")
K=3
def expand(Q_:str, vocabulary):
    if not os.path.exists(SIM_PATH):
        return Q_
    Q_in = [w for w in Q_.split() if w in vocabulary.tolist()]
    indexes = [np.where(pd.Index(pd.DataFrame(vocabulary)) == w)[0][0] for w in Q_in]
    max_sim_max = []
    sim = np.load(SIM_PATH)
    sim_ = np.nan_to_num(sim[indexes, :])
    for i, index in enumerate(indexes):
        sim_[i, index] = 0
    del sim
    index = sim_.argsort(axis=1)[:, -K:]
    sim2 = np.zeros_like(sim_)
    for i in range(index.shape[0]):
        sim2[i, index[i, :]] = sim_[i, index[i, :]]
    try:
        Q_in.extend(vocabulary[(sim2.sum(axis=0) / K).argsort()[-K:]].tolist())
    except IndexError:
        return Q_
    return " ".join(Q_in)
