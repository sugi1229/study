import numpy as np
from numpy.random import *

NPZ_PATH = 'res.npz'

def load_npz(path):
    a = np.load(path)
    out = a['result']
    name = a['name']
    return out, name

def calc_sim(out, name):
    dot_all = np.dot(out, out.T)
    norm_all = np.linalg.norm(out, axis=1)
    norm_all = norm_all[:,np.newaxis]
    norm = np.dot(norm_all, norm_all.T)
    all_score = dot_all/norm

    return all_score

out_model, name_model = load_npz(NPZ_PATH)
f_model = name_model.tolist()

data_model = calc_sim(out_model, name_model)
np.savez_compressed('cos_res.npz', result=data_model, name=name_model)

