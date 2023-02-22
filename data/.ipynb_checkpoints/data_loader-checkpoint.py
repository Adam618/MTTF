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

class Dataset_Custom(Dataset):
    def __init__(self,valid_data_path,data_division,lag_step,is_img_embed, img_channel,root_img_path, img_path, root_path,img_pred_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
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
        type_map = {'train':0, 'val':1, 'test':2}
        self.data_division = data_division
        self.set_type = type_map[flag]
        self.lag_step = lag_step
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_img_path = root_img_path
        self.img_path = img_path
        self.img_channel = img_channel
        self.root_path = root_path
        self.data_path = data_path
        self.is_img_embed = is_img_embed
        self.img_pred_len = img_pred_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        num_train = int(len(df_raw)*self.data_division[0])
        num_test = int(len(df_raw)*self.data_division[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        print(border1s)
        print(border2s)
       
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            # normalization
            self.mean, self.std = self.scaler.fit(train_data.values)
            print(self.mean)
            print(self.std)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        time = df_raw.date.apply(lambda x: pd.to_datetime(x).strftime('%Y_%m_%d_%H')).values
        # time encoder
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # through time_x read qixiang data
        self.time_x = time[border1:border2]

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

        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        img_name = self.time_x[s_begin-self.lag_step:s_end-self.lag_step]
        PATH_PREFIX = os.path.join(self.root_img_path,
                     self.img_path)
        # img_pred_len
        img_data = []
        # if add img data
        if self.is_img_embed:
            for i in range(img_name.shape[0]):
                path = os.path.join(PATH_PREFIX, img_name[i]+'.npy')
                t = np.load(path)
                img_data.append(t)
            img_data = np.array(img_data)
            
             
            if self.img_channel == 1:
                img_data = img_data[:, np.newaxis, ::]
                
            if self.scale:
                     # t,q
                # img_mean = np.array([0.003167965024205165,-3.34949108091085277])
                # img_std = np.array([0.0025393404120686593,6.139398799826309])
                img_mean = np.array([-11.162050158884638,0.0016886828817535117])
                img_std = np.array([7.466154578616134,0.0017431325336688677])
                img_data_t = (img_data[:,0,:,:]-img_mean[0])/img_std[0]
                img_data_t = img_data_t[:, np.newaxis, ::]
                img_data_q = (img_data[:,1,:,:]-img_mean[1])/img_std[1]
                img_data_q = img_data_q[:, np.newaxis, ::]
                img_data = np.concatenate((img_data_t,img_data_q),-3)
                # img_data = img_data_t
                # cuona q=t !!!!
                # print(img_data.shape)
           
        return seq_x, seq_y, seq_x_mark, seq_y_mark, img_data
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, valid_data_path,data_division, lag_step,is_img_embed, img_channel, root_img_path, img_path, root_path, img_pred_len, flag='train',
                 size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        self.valid_data_path = valid_data_path
        self.data_division = data_division
        self.lag_step = lag_step
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_img_path = root_img_path
        self.img_path = img_path
        self.img_channel = img_channel
        self.root_path = root_path
        self.data_path = data_path
        self.is_img_embed = is_img_embed
        self.img_pred_len = img_pred_len
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.valid_data_path))


        df_raw = df_raw[32-self.seq_len:]
        print(32-self.label_len)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            df_train = pd.read_csv(os.path.join(self.root_path,
                                                self.data_path))
            df_train = df_train[[self.target]]
            num_train = int(len(df_train) * self.data_division[0])
            df_train = df_train[:num_train]
            # self.scaler.mean = np.array([1.78065666])
            # self.scaler.std = np.array([11.05309861])
            self.scaler.fit(df_train.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # past
        tmp_stamp = df_raw[['date']]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # future
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq='3h')
        # past + future
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        time = df_stamp.date.apply(lambda x: pd.to_datetime(x).strftime('%Y_%m_%d_%H')).values
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])
        self.data_x = data
        # time encoder
        # through time to get modality data
        self.time = time
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        # seq_y = label
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        img_name = self.time[r_begin-self.lag_step:s_end + self.img_pred_len-self.lag_step]
        PATH_PREFIX = os.path.join(self.root_img_path,
                                   self.img_path)
        # img_pred_len
        img_data = []
        # if add img data
        if self.is_img_embed:
            for i in range(img_name.shape[0]):
                path = os.path.join(PATH_PREFIX, img_name[i] + '.npy')
                t = np.load(path)
                img_data.append(t)
            img_data = np.array(img_data)
            if self.img_channel == 1:
                img_data = img_data[:, np.newaxis, ::]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, img_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



