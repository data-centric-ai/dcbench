""" All sort and count algorithms

sort_count_no_dp(S, y, K)
sort_count_dp(S, y, K)
sort_count_dpdc(S, y, K)
sort_count_dpdc_after_clean(S, y, K)
sort_count_after_clean(S, y, K)
"""
import numpy as np
import pandas as pd
import time
from collections import Counter
from itertools import combinations
import collections
import itertools
from copy import deepcopy
from ..utils import Pool
from functools import partial
from .utils import compute_entropy_by_counts

def sort(S, y):
    """Squash similarity matrix in an array and sort by similarity ascendingly
        
    Args:
        S (list of list): similarity matrix
        y (np.array): labels
    
    Return:
        A (list of tuple): [(sij, ri, rj, yi)]
    """
    A = []
    for ri, (Si, yi) in enumerate(zip(S, y)):
        for rj, sij in enumerate(Si):
            A.append((-sij, ri, rj, yi))

    # break tie: small index has larger similarity 
    A = sorted(A)[::-1]
    sorted_A = []

    for sij, ri, rj, yi in A:
        sorted_A.append((-sij, ri, rj, yi))
    return sorted_A 

def compute_B(alpha_beta, K, eps=1e-100):
    n = len(alpha_beta)
    B = np.zeros((n+1, K+1))
    B[n, 0] = 1
    s = 1
    status = "big"

    for i in reversed(range(n)):
        b_b = alpha_beta[i][1] * B[i+1]
        B[i] = alpha_beta[i][0] * B[i+1]
        B[i, 1:] += b_b[:-1]
        s -= B[i+1, -1] * alpha_beta[i][1]

        if s < eps/n:
            status = "small"
            break

    return B, status

def compute_BR(alpha_beta, K, eps=1e-100):
    n = len(alpha_beta)
    B = np.zeros((n+1, K+1))
    B[0, 0] = 1
    s = 1
    status = "big"

    for i in range(1, n+1):
        b_b = alpha_beta[i-1][1] * B[i-1]
        B[i] = alpha_beta[i-1][0] * B[i-1]
        B[i, 1:] += b_b[:-1]
        s -= B[i-1, -1] * alpha_beta[i-1][1]

        if s < eps / n:
            status = "small"
            break
    return B, status

def group_by_classes(S, y):
    classes = [0, 1]

    # map old row number to new row number in each group
    new_rid = {}

    # number of rows in each group
    N_c = {c:0 for c in classes}

    # number of candidates for each row
    row_count = {c:[] for c in classes}

    # initialze alpha beta counters grouped by classes
    alpha_beta_c = {c:[] for c in classes}

    for ri, (Si, yi) in enumerate(zip(S, y)):
        new_rid[ri] = len(alpha_beta_c[yi]) #new row number
        row_count[yi].append(len(Si))
        alpha_beta_c[yi].append([0, 1])
        N_c[yi] += 1

    n_must_alpha_c = {c:0 for c in classes}
    n_must_beta_c = {c:N_c[c] for c in classes}
    return alpha_beta_c, new_rid, classes, N_c, row_count, n_must_alpha_c, n_must_beta_c

def change_alpha_beta(alpha_beta_c, n_must_alpha_c, n_must_beta_c, ri, yi, ab_after):
    """ Update alpha beta counters and n_must_alpha, n_must_beta
        Used for temporily changing alpha beta counters
    """
    ab_prev = alpha_beta_c[yi][ri]
    n_must_a_prev = 1 if ab_prev[1] == 0 else 0
    n_must_b_prev = 1 if ab_prev[0] == 0 else 0
    n_must_a_after = 1 if ab_after[1] == 0 else 0
    n_must_b_after = 1 if ab_after[0] == 0 else 0
    n_must_alpha_c[yi] += (n_must_a_after - n_must_a_prev)
    n_must_beta_c[yi] += (n_must_b_after - n_must_b_prev)
    alpha_beta_c[yi][ri] = ab_after

def get_cases(n_must_alpha_c, n_must_beta_c, N_c, classes, K):
    cases_c = {}

    for c in classes:
        n_must_alpha = n_must_alpha_c[c]
        n_must_beta = n_must_beta_c[c]
        N = N_c[c]
        cases_c[c] = set(range(n_must_beta, min(N-n_must_alpha+1, K+1)))

    ####Only work for binary cases
    possible_cases = []
    max_n_beta = {c:0 for c in classes}
    classes = list(classes)

    for i in range(K+1):
        if i in cases_c[classes[0]] and (K-i) in cases_c[classes[1]]:
            pred = classes[0] if i > K / 2 else classes[1]
            possible_cases.append({classes[0]:i, classes[1]:K-i, "knn_pred":pred})
            max_n_beta[classes[0]] = max(max_n_beta[classes[0]], i)
            max_n_beta[classes[1]] = max(max_n_beta[classes[1]], K-i)

    return possible_cases, max_n_beta

def sort_count_dp(S_full, y_full, K, mm=None):
    S, y, valid_indices = prune(S_full, y_full, K, mm)

    N = len(S)

    alpha_beta_c, new_rid, classes, N_c, row_count, n_must_alpha_c, n_must_beta_c = group_by_classes(S, y)
    world_counts = {c: 0 for c in classes}
    world_ub = {c:1 for c in classes}

    # sort
    sorted_A = sort(S, y)    

    # scan and count
    Bc = {}

    for sij, ri, rj, yi in sorted_A:
        new_ri = new_rid[ri]

        # temporarily change alpha beta for current row
        temp_ab = alpha_beta_c[yi][new_ri]
        change_alpha_beta(alpha_beta_c, n_must_alpha_c, n_must_beta_c, new_ri, yi, [0, 1/row_count[yi][new_ri]])

        # get possible cases
        cases, max_n_beta = get_cases(n_must_alpha_c, n_must_beta_c, N_c, classes, K)

        if len(cases) > 0:
            # compute Bc
            for c in classes:
                Bc[c], status = compute_B(alpha_beta_c[c], max_n_beta[c])
                if status == "small":
                    break

            if status == "big":
                for case in cases:
                    counts = np.prod([Bc[c][0, case[c]] for c in classes])
                    world_counts[case["knn_pred"]] += counts

        # reset alpha beta
        new_ab = [temp_ab[0] + 1 / row_count[yi][new_ri], temp_ab[1] - 1 / row_count[yi][new_ri]]
        new_ab = stablelize(new_ab)
        change_alpha_beta(alpha_beta_c, n_must_alpha_c, n_must_beta_c, new_ri, yi, new_ab)
    return world_counts

def init_counter(classes):
    return {c:0 for c in classes}

def init_ac_counters(S, classes):
    ac_counters = []
    for Si in S:
        counters = [init_counter(classes) for _ in range(len(Si))]
        ac_counters.append(counters)
    return ac_counters

def count_worlds_fix_ri(B, BR, n_beta, ri):
    if n_beta < 0:
        return 0
    if ri == len(B)-2:
        # the bottom row, only use BR
        n_world = BR[ri, n_beta] 
    elif ri == 0:
        # the top row, only use B
        n_world = B[1, n_beta]
    else:
        # n_world = sum([B[ri+1, int(l)] * BR[ri, int(n_beta-l)] for l in range(n_beta+1)])
        n_world = B[ri+1, :n_beta+1].dot(BR[ri, :n_beta+1][::-1])
    return n_world

def count_worlds_after_clean(B_c, BR_c, cases, classes, y, new_rid, dirty_rows):
    count_worlds = []

    for ri, yi in enumerate(y):
        if ri not in dirty_rows:
            # omit clean rows
            count_worlds.append([None, None])
            continue

        # initialize counter
        l_count = {c:0 for c in classes}
        s_count = {c:0 for c in classes}

        new_ri = new_rid[ri]
        other_c = [c for c in classes if c != yi][0] # classes other than the current one

        for case in cases:
            # worlds for other classes
            world_other = B_c[other_c][0, case[other_c]]

            # world for the current class
            n_beta_yi = case[yi]
            world_yi_s = count_worlds_fix_ri(B_c[yi], BR_c[yi], n_beta_yi, new_ri)
            world_yi_l = count_worlds_fix_ri(B_c[yi], BR_c[yi], n_beta_yi-1, new_ri)

            pred = case["knn_pred"]
            s_count[pred] += world_yi_s * world_other
            l_count[pred] += world_yi_l * world_other

        count_worlds.append([s_count, l_count])
    return count_worlds

def update_ac_counters(ac_counters, sl_counts, ri, rj, S, y, dirty_rows):
    # update current element using large counts
    if ri in dirty_rows:
        l_count = sl_counts[ri][1]
        for c, value in l_count.items():
            ac_counters[ri][rj][c] += value

    # update others
    for i, Si in enumerate(S):
        if i == ri:
            continue

        if i not in dirty_rows:
            continue

        s_count, l_count = sl_counts[i]
        for j, sij in enumerate(Si):
            if sij < S[ri][rj] or (sij == S[ri][rj] and i > ri):
                for c, value in s_count.items():
                    ac_counters[i][j][c] += value
            else:
                for c, value in l_count.items():
                    ac_counters[i][j][c] += value
    return ac_counters

def prune(S_full, y_full, K, mm=None):
    if mm is None:
        mm = np.array([[min(s), max(s)] for s in S_full])

    valid_indices = get_valid_indices(mm, K)

    S = [S_full[i] for i in valid_indices]
    y = [y_full[i] for i in valid_indices]
    return S, y, valid_indices

def get_valid_indices(mm, K):
    min_order = np.argsort(-mm[:, 0], kind="stable")
    k_largest_min_idx = min_order[K-1]
    k_largest_min = mm[k_largest_min_idx, 0]
    
    is_valid = np.zeros(len(mm))
    is_valid[:k_largest_min_idx+1] = (mm[:k_largest_min_idx+1, 1] >= k_largest_min)
    is_valid[k_largest_min_idx+1:] = (mm[k_largest_min_idx+1:, 1] > k_largest_min)
    valid_indices = np.argwhere(is_valid).ravel()
    return valid_indices

def compute_after_entropy(valid_indices, y_full, ac_counters, dirty_rows):
    after_entropies = []
    indices_map = {idx: i for i, idx in enumerate(valid_indices)}

    for i in range(len(y_full)):
        if i not in indices_map:
            after_entropies.append(None)
        elif indices_map[i] not in dirty_rows:
            after_entropies.append(None)
        else:
            entropies = [compute_entropy_by_counts(counts) for counts in ac_counters[indices_map[i]]]
            after_entropies.append(entropies)
    return after_entropies

def sort_count_after_clean(S_full, y_full, K, mm=None):
    tic = time.time()
    S, y, valid_indices = prune(S_full, y_full, K, mm)

    # print("prune", len(S), time.time() - tic)
    # omit clean rows in later computation
    dirty_rows = set([i for i, x in enumerate(S) if len(x) > 1])

    # initialize
    alpha_beta_c, new_rid, classes, N_c, row_count, n_must_alpha_c, n_must_beta_c = group_by_classes(S, y)
    ac_counters = init_ac_counters(S, classes)

    # sort
    sorted_A = sort(S, y)

    # dp tables
    B_c = {}
    BR_c = {}
    # scan
    for sij, ri, rj, yi in sorted_A:
        new_ri = new_rid[ri]

        # temporarily change alpha beta to [0, 1] for current row
        temp_ab = alpha_beta_c[yi][new_ri]
        change_alpha_beta(alpha_beta_c, n_must_alpha_c, n_must_beta_c, new_ri, yi, [0, 1/row_count[yi][new_ri]])

        # get possible cases
        cases, max_n_beta = get_cases(n_must_alpha_c, n_must_beta_c, N_c, classes, K)

        # skip if no possible cases
        if len(cases) > 0:
            # compute dp tables B_c, BR_c only for the class of the current element
            for c in classes:
                B_c[c], status = compute_B(alpha_beta_c[c], max_n_beta[c])
                if status == "small":
                    break
                BR_c[c], status = compute_BR(alpha_beta_c[c], max_n_beta[c])
                if status == "small":
                    break

            if status == "big":
                # count worlds for each cell
                sl_counts = count_worlds_after_clean(B_c, BR_c, cases, classes, y, new_rid, dirty_rows)

                # update counts
                ac_counters = update_ac_counters(ac_counters, sl_counts, ri, rj, S, y, dirty_rows)
            
        # restore and update alpha beta
        new_ab = [temp_ab[0] + 1 / row_count[yi][new_ri], temp_ab[1] - 1 / row_count[yi][new_ri]]
        new_ab = stablelize(new_ab)
        change_alpha_beta(alpha_beta_c, n_must_alpha_c, n_must_beta_c, new_ri, yi, new_ab)

    after_entropies = compute_after_entropy(valid_indices, y_full, ac_counters, dirty_rows)
    return after_entropies

def stablelize(new_ab):
    if new_ab[0] < 1e-9:
        new_ab = [0, 1]
    if new_ab[1] < 1e-9:
        new_ab = [1, 0]
    return new_ab

def sort_count_dp_wrapper(i, S_val, y_train, K, MM):
    return sort_count_dp(S_val[i], y_train, K, mm=MM[i])

def sort_count_dp_multi(S_val, y_train, K, MM=None, n_jobs=4):
    indices = range(len(S_val))
    if MM is None:
        MM = [None] * len(S_val)

    if n_jobs == 1:
        counters = [sort_count_dp(S_val[i], y_train, K, MM[i]) for i in indices]
    else:
        pool = Pool(n_jobs)    
        counters = pool.map(partial(sort_count_dp_wrapper, S_val=S_val, y_train=y_train, K=K, MM=MM), indices)
    return counters

def sort_count_after_clean_wrapper(i, S_val, y_train, K, MM):
    return sort_count_after_clean(S_val[i], y_train, K, mm=MM[i])

def sort_count_after_clean_multi(S_val, y_train, K, n_jobs=4, MM=None):
    indices = range(len(S_val))
    if MM is None:
        MM = [None] * len(S_val)

    if n_jobs == 1:
        after_entropies = [sort_count_after_clean(S_val[i], y_train, K, MM[i]) for i in indices]
    else:
        pool = Pool(n_jobs)    
        after_entropies = pool.map(partial(sort_count_after_clean_wrapper, S_val=S_val, y_train=y_train, K=K, MM=MM), indices)
    return after_entropies

if __name__ == '__main__':
    # S = np.array([[1, 1.5, 2.1], [2.2, 2.4, 3], [1.6, 2.3, 2.5], [1.1, 3.5, 4], [1.2, 4.5, 5]])
    # y = np.array([1, 0, 0, 1, 1])
    np.random.seed(1)
    N = 5000
    K = 3
    S = np.random.rand(N, 10) 
    y = np.random.randint(0, 2, size=N)

    tic = time.time()
    result = sort_count_after_clean(S, y, K, np.arange(N))
    print(time.time() - tic)
    # print(result)

    tic = time.time()
    old_result = sort_count_after_clean_old(S, y, K, np.arange(N))
    print(time.time() - tic)
    # print(old_result)
    # tic = time.time()
    # ae1 = sort_count_after_clean(S, y, 3)
    # print(time.time() - tic, ae1[0] / (ae1[1] + ae1[0]))


    

    # print()
    # N = 100
    # alpha = np.random.randint(1, 10, size=(N, ))
    # beta = 10 - alpha

    # tic = time.time()
    # B = compute_BR(alpha/10, beta/10, 3)
    # print(time.time() - tic)
    # print(B[0][-1])

    # alpha_beta = np.vstack([alpha, beta]).T.tolist()

    # tic = time.time()
    # B_old = compute_BR_old(alpha_beta, 3)
    # print(max(B_old[-1]) / (10 ** 100))
    # print(time.time() - tic)



    # print(sort_count_entropy(S, y, 3))


#     tic = time.time()
#     from copy import deepcopy
#     S = S.tolist()
#     ae2 = []
#     for i in range(5):
#         ae_i = []
#         for j in range(3):
#             S_a = deepcopy(S)
#             S_a[i] = [S[i][j]]
#             count = sort_count_dpdc(S_a, y, 3)
#             ae_i.append(count)
#         ae2.append(ae_i)
#     print(time.time() - tic)

#     for a1, a2 in zip(ae1, ae2):
#         for e1, e2 in zip(a1, a2):
#             print(e1 == e2)

    # print(sort_count_no_dp(S, y, 1))

    # print(sort_count_entropy(S, y, 3))
    
    # S = np.array([[1, 1.5, 2.1], [2.2, 2.4, 3.1], [1.6, 2.3, 2.5], [1.1, 3.5, 4], [4.1, 4.5, 5]])
    # y = np.array([1, 0, 0, 1, 1])
    # print(sort_count_no_dp(S, y, 3))
    # print("___________")
    # print(sort_count_dp(S, y, 3))
    # print(sort_count_dpdc(S, y, 3))

    # N = 1000
    # S = np.random.rand(N, 10) 
    # y = np.random.randint(0, 2, size=N)
    # tic = time.time()
    # print(sort_count_entropy(S, y, 3))
    # print(time.time() - tic)


    # a = {i:10 for i in range(10000)}
    # b = {i:10 for i in range(5000, 15000)}
    # tic = time.time()
    # c_dict = dict_product3(a, b)
    # print(len(a))
    # print(len(c_dict))
    # print(time.time() - tic)

    # N = 10000
    # alpha = np.random.randint(1, 10, size=(N, )).tolist()
    # beta = np.random.randint(1, 10, size=(N, )).tolist()
    
    # tic = time.time()
    # B = compute_B(alpha, beta, 3)
    # BR = compute_BR(alpha, beta, 3)

    # print(B[0][2] == BR[N][2])
    # print((time.time() - tic)*N)