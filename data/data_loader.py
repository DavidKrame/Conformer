import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len,
                    12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len,
                    12*30*24*4+1*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+1*30*24*4, 12*30*24*4+2*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_elec(Dataset):
    def __init__(self, data_set, root_path, flag='train', size=None,
                 features='MS', data_path='elec12.csv',
                 target='target', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 8
            self.pred_len = 8
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data_set = data_set
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        """
        df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y-%m-%d %H:%M:%S')
        train_start_index = df_raw[df_raw["date"]== self.train_start_date].index.tolist()[0]
        train_end_index = df_raw[df_raw["date"]== self.train_end_date].index.tolist()[0]
        val_start_index = df_raw[df_raw["date"]== self.val_start_date].index.tolist()[0]
        val_end_index = df_raw[df_raw["date"]== self.val_end_date].index.tolist()[0]
        test_start_index = df_raw[df_raw["date"] == self.test_start_date].index.tolist()[0]
        test_end_index = df_raw[df_raw["date"] == self.test_end_date].index.tolist()[0]
        pred_start_index = df_raw[df_raw["date"]== self.pred_start_date].index.tolist()[0]
        pred_end_index = df_raw[df_raw["date"]== self.pred_end_date].index.tolist()[0]

        border1s = [train_start_index, val_start_index-self.seq_len,
                    test_start_index-self.seq_len, pred_start_index-self.seq_len]#35137
        border2s = [train_end_index, val_end_index+self.seq_len-1,
                    test_end_index+self.seq_len-1, pred_end_index+1]
        """
        #border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+2*30*24*4 - self.seq_len]
        #border2s = [12*30*24*4, 12*30*24*4+2*30*24*4, 12*30*24*4+4*30*24*4]
        if self.data_set == 'elec':  # 12/2/2
            border1s = [0, 12*30*24*4 - self.seq_len,
                        12*30*24*4+1*30*24*4 - self.seq_len]
            border2s = [12*30*24*4, 12*30*24*4+1*30*24*4, 12*30*24*4+2*30*24*4]
        elif self.data_set == 'ECL':  # 7/1/1
            # border1s = [0, 12*30*24 - self.seq_len,
            #             12*30*24+2*30*24 - self.seq_len]
            # border2s = [12*30*24, 12*30*24+2*30*24, 12*30*24+4*30*24]
            border1s = [0, 12*30*24*4 - self.seq_len,
                        12*30*24*4+2*30*24*4 - self.seq_len]
            border2s = [12*30*24*4, 12*30*24*4+2*30*24*4, 12*30*24*4+4*30*24*4]
        elif self.data_set == 'WTH':  # 24/2/2
            # border1s = [0, 24*30*6*8 - self.seq_len, 24*30*6*8+6*30*24 - self.seq_len]
            # border2s = [24*30*6*8, 24*30*6*8+30*24*6, 24*30*6*8+6*30*24*2]
            border1s = [0, 7*30*24*6 - self.seq_len,
                        7*30*24*6+2*30*24*6 - self.seq_len]
            border2s = [7*30*24*6, 7*30*24*6+2*30*24*6, 7*30*24*6+4*30*24*6]
        elif self.data_set == 'TRAF':  # 15/3/3
            border1s = [0, 15*30*24 - self.seq_len,
                        15*30*24+3*30*24 - self.seq_len]
            border2s = [15*30*24, 15*30*24+3*30*24, 15*30*24+6*30*24]
        elif self.data_set == 'ETTm1':  # 12/2/2
            border1s = [0, 12*30*24*4 - self.seq_len,
                        12*30*24*4+1*30*24*4 - self.seq_len]
            border2s = [12*30*24*4, 12*30*24*4+1*30*24*4, 12*30*24*4+2*30*24*4]
        elif self.data_set == 'ETTh1':  # 12/2/2
            border1s = [0, 12*30*24 - self.seq_len,
                        12*30*24+2*30*24 - self.seq_len]
            border2s = [12*30*24, 12*30*24+2*30*24, 12*30*24+4*30*24]
            # border1s = [0, 12*30*24*4 - self.seq_len,
            #             12*30*24*4+1*30*24*4 - self.seq_len]
            # border2s = [12*30*24*4, 12*30*24*4+1*30*24*4, 12*30*24*4+2*30*24*4]
        else:
            border1s = [0, 16*365 - self.seq_len, 16*365+2*365 - self.seq_len]
            border2s = [16*365, 16*365+2*365, 16*365+4*365]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            #df_data = df_raw[['pred_w_speed', 'pred_w_dir', 'pred_temp', 'pred_humidity', self.target]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type >= 3:
            index = index * self.pred_len
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_in = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_in

    def __len__(self):
        if self.set_type < 3:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        return int((len(self.data_x) - self.seq_len)/96)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_airdelay(Dataset):
    def __init__(self, data_set, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',  cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.dropna(axis=0, how='any')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('Cancelled')
        #df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        input_point = len(df_raw)
        num_train = int(input_point * 0.7)
        num_test = int(input_point * 0.2)
        #num_vali = len(df_raw) - num_train - num_test
        num_vali = int(input_point * 0.1)
        border1s = [0, num_train - self.seq_len,
                    num_train + num_vali-self.seq_len]
        border2s = [num_train, num_train + num_vali,
                    num_train + num_vali + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            #cols_data = df_raw.columns[2:]
            df_data = df_raw[cols[2:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_raw['arrTime'] = pd.DataFrame(
            (df_raw['arrTime'].apply(int)).apply(str).str.zfill(4))
        df_raw['arrTime'] = df_raw['arrTime'].replace('2400', '2359')
        # print(len(df_raw['arrTime']))
        tmp = df_raw['FlightDate'].add(' ').add(df_raw['arrTime'])
        # print(tmp[1540:1542])
        df_stamp = pd.to_datetime(tmp, format='%Y/%m/%d %H%M')

        #df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # pd.to_datetime(df_raw['FlightDate'])
        #df_stamp = pd.date_range(start='1/1/2022',periods=border2-border1, freq='M')
        #df_stamp = df_raw[['date']][border1:border2]
        #df_stamp['date'] = pd.to_datetime(df_stamp.date)
        #df_stamp['date'] = df_stamp
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        #self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'test': 1, 'val': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            #p = list(df_raw.columns)[0][0]
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('Date Time')
        df_raw = df_raw[[p]+cols+[self.target]]
        '''

        num_train = int(len(df_raw)*0.8)
        num_test = int(len(df_raw)*0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            #df_data = df_raw[['pred_w_speed', 'pred_w_dir', 'pred_temp', 'pred_humidity', self.target]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type >= 3:
            index = index * self.pred_len
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :-1]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_in = self.data_y[r_begin:r_end, :-1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_in

    def __len__(self):
        if self.set_type < 3:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        return int((len(self.data_x) - self.seq_len)/96)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
