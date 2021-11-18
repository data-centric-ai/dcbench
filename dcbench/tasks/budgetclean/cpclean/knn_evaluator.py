import numpy as np

def compute_distances(X_train, X_test):
    dists = np.array([np.sqrt(np.sum((X_train - x_test)**2, axis=1)) for x_test in X_test])
    return dists

def majority_vote(A):
    major = np.argmax(np.bincount(A))
    return int(major)

class KNNEvaluator(object):
    """docstring for KNNEvaluator"""
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, K=3):
        super(KNNEvaluator).__init__()
        dists_val = compute_distances(X_train, X_val)
        dists_test = compute_distances(X_train, X_test)
        self.sim_val = 1 / (1 + dists_val)
        self.sim_test = 1 / (1 + dists_test)
        self.K = K
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def predict(self, sim):
        order = np.argsort(-sim, kind="stable", axis=1)
        top_K_idx = order[:, :self.K]
        top_K = self.y_train[top_K_idx]
        pred = np.array([majority_vote(top) for top in top_K])
        return pred

    def score(self):
        pred_val = self.predict(self.sim_val)
        pred_test = self.predict(self.sim_test)

        val_acc = (pred_val == self.y_val).mean()
        test_acc = (pred_test == self.y_test).mean()

        return val_acc, test_acc