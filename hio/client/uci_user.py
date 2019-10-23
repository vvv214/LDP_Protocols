import logging
import math
import os

import numpy as np
import pandas as pd

from client.user import User


class UCIUser(User):
    logging.basicConfig(format='%(levelname)s:%(asctime)s: - %(name)s - :%(message)s',
                        level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    def initial_generate(self):
        if self.args.user_type == 'bank':
            self.columns = [
                'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'duration', 'campaign',
                'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
                'nr.employed', 'income'
            ]
            self.num_column = [
                'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                'euribor3m', 'nr.employed',
            ]
            self.drop_column = [
                'pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m'
            ]
            self.ranges = {
                'age': {'min': 17, 'max': 81, 'gran': 1},
                'duration': {'min': 0, 'max': 4919, 'gran': 5},
                'previous': {'min': 0, 'max': 8, 'gran': 1},
                'campaign': {'min': 1, 'max': 64, 'gran': 1},
                # 'euribor3m': {'min': 0.634, 'max': 5.046, 'gran': 0.069},
                'nr.employed': {'min': 4963.6, 'max': 5228.2, 'gran': 4.14},
            }
            self.column_to_agg = 'previous'
            if self.args.perturb_type == 'cat1':
                self.column_to_perturb = ['job']
            elif self.args.perturb_type == 'ord1':
                self.column_to_perturb = ['age']
            elif self.args.perturb_type == 'ord1h':
                self.column_to_perturb = ['duration']
                # this line is used to test constraint inference, should comment out
                self.ranges['duration']['gran'] = np.round(
                    (self.ranges['duration']['max'] - self.ranges['duration']['min']) / (self.args.hie_fanout ** 4), 2)
            elif self.args.perturb_type == 'ord2':
                self.column_to_perturb = ['age', 'duration']
            elif self.args.perturb_type == 'ord2l':
                self.column_to_perturb = ['duration', 'nr.employed']
                self.ranges['duration']['gran'] = 20
                self.ranges['nr.employed']['gran'] = 1.04
            elif self.args.perturb_type == 'ord4':
                self.column_to_perturb = ['age', 'duration', 'campaign', 'nr.employed']
            elif self.args.perturb_type == 'ord2cat2':
                self.column_to_perturb = ['age', 'duration', 'job', 'education']
        elif self.args.user_type == 'adult':
            self.columns = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            self.num_column = [
                'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
            ]
            self.drop_column = ['fnlwgt', ]
            self.ranges = {
                'age': {'min': 17, 'max': 81, 'gran': 1},
                'education-num': {'min': 1, 'max': 16, 'gran': 1},
                'capital-gain': {'min': 0, 'max': 99999, 'gran': 98},
                'capital-loss': {'min': 0, 'max': 6400, 'gran': 100},
                'hours-per-week': {'min': 1, 'max': 65, 'gran': 1},
            }
            self.column_to_agg = 'capital-gain'
            self.column_to_agg = 'education-num'
            if self.args.perturb_type == 'cat1':
                self.column_to_perturb = ['native-country']
            elif self.args.perturb_type == 'ord1':
                self.column_to_perturb = ['hours-per-week']
            elif self.args.perturb_type == 'ord1h':
                self.column_to_perturb = ['capital-gain']
            elif self.args.perturb_type == 'ord2':
                self.column_to_perturb = ['age', 'capital-gain']
            elif self.args.perturb_type == 'ord2l':
                self.column_to_perturb = ['capital-gain', 'capital-loss']
                self.ranges['capital-gain']['gran'] = 391
                self.ranges['capital-loss']['gran'] = 40
            elif self.args.perturb_type == 'ord4':
                self.column_to_perturb = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
            elif self.args.perturb_type == 'ord2cat2':
                self.column_to_perturb = ['age', 'sex', 'hours-per-week', 'native-country']
        else:
            self.columns = [
                "YEAR", "DATANUM", "SERIAL", "HHWT", "GQ", "PERNUM", "PERWT", "AGE", "MARST", "RACE", "RACED",
                "EDUC", "EDUCD",
                "DEGFIELD", "DEGFIELDD", "UHRSWORK", "INCTOT", "FTOTINC", "INCWAGE", "INCSS"
            ]
            self.num_column = [
                'AGE', 'UHRSWORK', 'INCTOT', 'FTOTINC', 'INCWAGE', 'INCSS',
            ]
            self.drop_column = ['YEAR', 'DATANUM', 'SERIAL', 'HHWT', 'PERNUM', 'PERWT',
                                'RACED', 'EDUCD', 'DEGFIELDD', ]
            self.ranges = {
                'AGE': {'min': 17, 'max': 81, 'gran': 1},
                'UHRSWORK': {'min': 0, 'max': 99, 'gran': 10},
                'INCTOT': {'min': 0, 'max': 9999999, 'gran': 9766},
                'FTOTINC': {'min': 0, 'max': 9999999, 'gran': 9766},
                'INCWAGE': {'min': 0, 'max': 999999, 'gran': 977},
                'INCSS': {'min': 0, 'max': 99999, 'gran': 1563},
            }
            self.column_to_agg = 'UHRSWORK'
            if self.args.perturb_type == 'cat1':
                self.column_to_perturb = ['DEGFIELDD']
            elif self.args.perturb_type == 'ord1':
                self.column_to_perturb = ['AGE']
            elif self.args.perturb_type in ['ord1h', 'ord1hU']:
                self.column_to_perturb = ['INCTOT']
            elif self.args.perturb_type == 'ord2':
                self.column_to_perturb = ['AGE', 'INCTOT']
            elif self.args.perturb_type == 'ord3':
                self.column_to_perturb = ['AGE', 'INCTOT', 'FTOTINC']
                self.ranges['INCTOT']['gran'] = 39063
                self.ranges['FTOTINC']['gran'] = 39063
            elif self.args.perturb_type == 'ord3l':
                self.column_to_perturb = ['INCTOT', 'FTOTINC', 'INCWAGE']
                self.ranges['INCTOT']['gran'] = 39063
                self.ranges['FTOTINC']['gran'] = 39063
                self.ranges['INCWAGE']['gran'] = 39063
            elif self.args.perturb_type == 'ord3q2':
                self.column_to_perturb = ['AGE', 'INCTOT', 'FTOTINC']
                self.ranges['INCTOT']['gran'] = 39063
                self.ranges['FTOTINC']['gran'] = 39063
            elif self.args.perturb_type == 'ord2l':
                self.column_to_perturb = ['INCTOT', 'FTOTINC']
                self.ranges['INCTOT']['gran'] = 39063
                self.ranges['FTOTINC']['gran'] = 39063
            elif self.args.perturb_type in ['ord1cat3q11', 'ord1cat3q12', 'ord1cat3q02', 'ord1cat3q13']:
                self.column_to_perturb = ['GQ', 'INCTOT', 'MARST', 'RACE', ]
            elif self.args.perturb_type in ['ord1lcat3q11', 'ord1lcat3q12', 'ord1lcat3q02', 'ord1lcat3q13']:
                self.column_to_perturb = ['GQ', 'INCTOT', 'MARST', 'RACE', ]
                self.ranges['INCTOT']['gran'] = 80000
            elif self.args.perturb_type == 'ord2cat2':
                self.column_to_perturb = ['GQ', 'AGE', 'MARST', 'INCTOT']
                self.ranges['INCTOT']['gran'] = 80000
                self.ranges['AGE'] = {'min': 0, 'max': 124, 'gran': 1}
            elif self.args.perturb_type in ['ord2cat2q10', 'ord2cat2q11', 'ord2cat2q20', 'ord2cat2q22', ]:
                self.column_to_perturb = ['GQ', 'MARST', 'INCTOT', 'FTOTINC']
                self.ranges['INCTOT']['gran'] = int(
                    math.ceil(self.ranges['INCTOT']['max'] / (5 ** self.args.num_domain)))
                self.ranges['FTOTINC']['gran'] = int(
                    math.ceil(self.ranges['FTOTINC']['max'] / (5 ** self.args.num_domain)))
            elif self.args.perturb_type in ['ord4cat4q01', 'ord4cat4q10', 'ord4cat4q02', 'ord4cat4q20', 'ord4cat4q11',
                                            'ord4cat4q12', 'ord4cat4q21']:
                self.column_to_perturb = ['GQ', 'AGE', 'MARST', 'RACE', 'EDUC', 'INCTOT', 'FTOTINC', 'INCSS', ]
                self.ranges['INCTOT']['gran'] = 80000
                self.ranges['AGE'] = {'min': 0, 'max': 124, 'gran': 1}
                self.ranges['FTOTINC']['gran'] = 80000
                self.ranges['INCSS']['gran'] = 8000
                if self.args.layer_style == 'al':
                    if self.args.perturb_type == 'ord4cat4q01':
                        self.column_to_perturb = ['MARST']
                    elif self.args.perturb_type == 'ord4cat4q02':
                        self.column_to_perturb = ['MARST', 'GQ']
                    elif self.args.perturb_type == 'ord4cat4q10':
                        self.column_to_perturb = ['AGE']
                    elif self.args.perturb_type == 'ord4cat4q10':
                        self.column_to_perturb = ['AGE']
                    elif self.args.perturb_type == 'ord4cat4q11':
                        self.column_to_perturb = ['INCTOT', 'GQ']
                    elif self.args.perturb_type == 'ord4cat4q12':
                        self.column_to_perturb = ['INCTOT', 'GQ', 'MARST']
                    elif self.args.perturb_type == 'ord4cat4q20':
                        self.column_to_perturb = ['INCTOT', 'AGE']
                    elif self.args.perturb_type == 'ord4cat4q20':
                        self.column_to_perturb = ['INCTOT', 'AGE', 'INCTOT']
                # self.ranges['INCTOT']['gran'] *= 5
                # self.ranges['AGE']['gran'] *= 5
                # self.ranges['FTOTINC']['gran'] *= 5
                # self.ranges['INCSS']['gran'] *= 5

        df = self.get_data('./data/csvs')
        X = self.transform(df)
        if self.args.dup_rate > 1:
            X = np.repeat(X, self.args.dup_rate, axis=0)
        np.random.shuffle(X)
        self.X_all = self.X = X

        logging.info('finish transform, number of features: %d ' % len(self.X[0]))

    def get_data(self, df_path):
        user_type = self.args.user_type
        if self.args.user_type == 'ipums500k' or self.args.user_type == 'ipums2m' or self.args.user_type == 'ipums1m':
            user_type = 'ipums'
        logging.info('loading %s..' % user_type)
        df = pd.read_csv(os.path.join(df_path, '%s.all.txt' % user_type), header=None, names=self.columns, na_values='?')
        # df = pd.read_csv(os.path.join(df_path, '%s.txt' % user_type), header=None, names=self.columns, na_values="?")

        if self.args.user_type == 'ipums500k':
            if not self.args.vary == 'p':
                df = df.sample(n=500000, replace=False)
        elif self.args.user_type == 'ipums1m':
            df = df.sample(n=1000000, replace=False)

        for cat in self.columns:
            if df[cat].dtypes == 'object':
                df = df[df[cat] != ' ?']
        df = df.dropna(axis=1, how='any')
        return df

    def transform(self, df):
        if self.args.user_type == 'adult' or self.args.user_type == 'bank':
            if self.args.user_type == 'adult':
                df.replace({'income': {' <=50K.': ' <=50K', ' >50K.': ' >50K'}}, inplace=True)

            df.replace({'income': {' <=50K': '0', ' >50K': '1'}}, inplace=True)

        self.drop_column = list(
            set(self.columns).difference(set(self.column_to_perturb)).difference(set([self.column_to_agg])))
        df.drop(self.drop_column, axis=1, inplace=True)
        self.columns = [x for x in df.columns if x not in self.drop_column]

        # print stats
        # for col in self.columns:
        #     print(col, end=': ')
        #     if df[col].dtypes == 'object':
        #         print(df[col].unique())
        #     else:
        #         print(df[col].min(), df[col].max(), df[col].unique())

        for col in self.columns:
            bins = np.array([])
            if col not in self.ranges:
                df[col] = df[col].apply(str)
                df[col] = df[col].str.strip()
                df[col] = pd.Categorical(df[col])
            else:
                bins = np.round(np.arange(self.ranges[col]['min'], self.ranges[col]['max'], self.ranges[col]['gran']),
                                2)
                # too slow when there are many bins
                # df[col] = pd.cut(df[col], bins, right=False)
                start = bins[0]
                step = bins[1] - bins[0]
                if self.args.perturb_type in ['ord2cat2', 'ord2cat2q11', 'ord2cat2q20', 'ord2cat2q22', 'ord2cat2q10',
                                              'ord2cat2q12', 'ord4cat4q11', 'ord4cat4q12', 'ord4cat4q10',
                                              'ord4cat4q02', 'ord4cat4q01', 'ord4cat4q20', 'ord4cat4q21'] \
                        and col in ['INCTOT', 'FTOTINC'] and self.args.user_type[0:5] == 'ipums':
                    df[col] = np.random.randint(0, len(bins), df.shape[0])
                else:
                    df[col] = ((df[col].values - start) / step).astype(np.int16)

            logging.info('finish %s' % col)

            if bins.any():
                bins = [pd.Interval(x, np.round(x + self.ranges[col]['gran'], 2), closed='left') for x in bins]
                self.kv_map[col] = dict(enumerate(bins))
                self.vk_map[col] = {v: k for k, v in self.kv_map[col].items()}
            else:
                self.kv_map[col] = dict(enumerate(df[col].cat.categories))
                self.vk_map[col] = {v: k for k, v in self.kv_map[col].items()}
                df.replace({col: self.vk_map[col]}, inplace=True)

        return df.values.astype(np.int)
