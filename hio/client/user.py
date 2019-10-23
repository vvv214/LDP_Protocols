import numpy as np
import pandas as pd
import os.path
import itertools
from sklearn.datasets.samples_generator import make_blobs
import abc
import logging


class User(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        logging.basicConfig(format='%(levelname)s:%(asctime)s: - %(name)s - :%(message)s',
                            level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.args = args

        self.X = None
        self.X_all = None
        self.kv_map = {}
        self.vk_map = {}
        self.columns = []
        self.column_to_perturb = []
        self.column_to_agg = []
        self.drop_column = []
        self.num_column = []
        self.ranges = {}

        self.initial_generate()

        self.args.n = len(self.X)
        self.n = len(self.X)
        self.args.d = len(self.X[0])
        self.col_index_name_map = dict(enumerate(self.columns))
        self.col_name_index_map = {v: k for k, v in self.col_index_name_map.items()}

        self.colind_scolind_map = {}
        count = 0
        for col_name in self.column_to_perturb:
            self.colind_scolind_map[self.col_name_index_map[col_name]] = count
            count += 1


    @abc.abstractmethod
    def initial_generate(self):
        return

    def sample_p(self, p):
        self.X = self.X_all[np.random.choice(self.X_all.shape[0], int(self.X_all.shape[0] * p), replace=False), :]
        self.args.n = len(self.X)
        self.n = len(self.X)

    def sample_n(self, n):
        self.X = self.X_all[np.random.choice(self.X_all.shape[0], n, replace=False), :]
        self.args.n = len(self.X)
        self.n = len(self.X)
