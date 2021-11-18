import numpy as np
from copy import deepcopy
import pandas as pd
from .knn_evaluator import KNNEvaluator
import utils

class Debugger(object):
    """docstring for Debugger"""
    def __init__(self, data, model, debug_dir):
        self.data = deepcopy(data)
        self.K = model["params"]["n_neighbors"]
        self.debug_dir = debug_dir
        self.logging = []
        self.n_dirty = self.data["X_train_mv"].isnull().values.any(axis=1).sum()
        self.n_val = len(self.data["X_val"])

    def init_log(self, percent_cc):
        self.clean_val_acc, self.clean_test_acc = \
            KNNEvaluator(self.data["X_train_clean"], self.data["y_train"], 
                        self.data["X_val"], self.data["y_val"], 
                        self.data["X_test"], self.data["y_test"]).score()
        self.gt_val_acc, self.gt_test_acc = \
            KNNEvaluator(self.data["X_train_gt"], self.data["y_train"], 
                        self.data["X_val"], self.data["y_val"], 
                        self.data["X_test"], self.data["y_test"]).score()
        self.X_train_mean = deepcopy(self.data["X_train_repairs"]["mean"])
        self.selection = []

        self.logging = []
        mean_val_acc, mean_test_acc = \
            KNNEvaluator(self.X_train_mean, self.data["y_train"], 
                self.data["X_val"], self.data["y_val"], 
                self.data["X_test"], self.data["y_test"]).score()

        self.logging.append([0, self.n_val, None, None, percent_cc, 0, 
                             self.clean_val_acc, self.gt_val_acc, mean_val_acc,
                             self.clean_test_acc, self.gt_test_acc, mean_test_acc])
        self.save_log()

    def save_log(self):
        columns = ["n_iter", "n_val", "selection", "time", "percent_cc", "percent_clean", "clean_val_acc", 
                   "gt_val_acc", "mean_val_acc", "clean_test_acc", "gt_test_acc", "mean_test_acc"]
        logging_save = pd.DataFrame(self.logging, columns=columns)
        logging_save.to_csv(utils.makedir([self.debug_dir], "details.csv"), index=False)

    def log(self, n_iter, sel, sel_time, percent_cc):
        self.selection.append(sel)

        percent_clean = len(self.selection) / self.n_dirty
        self.X_train_mean[sel] = self.data["X_train_gt"][sel]

        mean_val_acc, mean_test_acc = KNNEvaluator(self.X_train_mean, self.data["y_train"], 
                                                self.data["X_val"], self.data["y_val"], 
                                                self.data["X_test"], self.data["y_test"]).score()

        self.logging.append([n_iter, self.n_val, sel, sel_time, percent_cc, percent_clean, 
                             self.clean_val_acc, self.gt_val_acc, mean_val_acc,
                             self.clean_test_acc, self.gt_test_acc, mean_test_acc])

        self.percent_clean = percent_clean
        self.save_log()



