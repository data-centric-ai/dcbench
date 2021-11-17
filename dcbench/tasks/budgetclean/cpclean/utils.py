import os
import pandas as pd
import numpy as np
import json
import pandas as pd
from scipy.stats import entropy
from multiprocessing import Process, Queue

def makedir(dir_list, file=None):
    save_dir = os.path.join(*dir_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

def dicts_to_csv(dicts, save_path):
    result = []
    for res in dicts:
        result.append(pd.Series(res))
    result = pd.concat(result).to_frame().transpose()
    result.to_csv(save_path, index=False)

def load_csv(save_dir):
    files = [f for f in os.listdir(save_dir) if f.endswith(".csv")]
    data = {}
    for f in files:
        name = f[:-4]
        data[name] = pd.read_csv(os.path.join(save_dir, f))
    return data

def load_cache(cache_dir):
    with open(os.path.join(cache_dir, "info.json"), "r") as f:
        info = json.load(f)

    data = load_csv(cache_dir)
    data["X_train_repairs"] = load_csv(os.path.join(cache_dir, "X_train_repairs"))
    return data, info

def compute_entropy(counts):
    """Compute entropy given counts of each label

    Args:
        counts (dict): {label: count}
    """
    s = sum(counts.values())
    p = [c/s for c in counts.values()]
    return entropy(p)

class Pool(object):
    """docstring for Pool"""
    def __init__(self, n_jobs):
        super(Pool, self).__init__()
        self.n_jobs = n_jobs

    def fn_batch(self, fn, arg_batch, q):
        res = [(i, fn(arg)) for i, arg in arg_batch]
        q.put(res)

    def array_split(self, arr, n):
        if len(arr) > n:
            res = []
            idx = np.array_split(np.arange(len(arr)), n)
            for i in idx:
                res.append([(j, arr[j]) for j in i])
        else:
            res = [[(i, a)] for i, a in enumerate(arr)]
        return res 

    def map(self, fn, args):
        arg_batches = self.array_split(args, self.n_jobs)

        q = Queue()
        procs = [Process(target=self.fn_batch, args=(fn, arg_batch, q)) for arg_batch in arg_batches]
        
        for p in procs:
            p.start()

        results = []
        for p in procs:
            results.extend(q.get())

        for p in procs:
            p.join()

        sorted_results = sorted(results)
        results = [res for i, res in sorted_results]
        return results