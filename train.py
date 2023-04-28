from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_elec, Dataset_airdelay
from exp.exp_basic import Exp_Basic
from models.model import Model

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.metrics import RMSE

import numpy as np
import math
import collections
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

def RandomVariable(sample, logp, entr, **params):
    return G.Struct(sample=sample, logp=logp, entr=entr, **params)

class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'Model':Model
        }
        if self.args.model=='Model' :
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len,
                self.args.step_len, 
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.normal_layers,
                self.args.enc_lstm,
                self.args.dec_lstm,
                self.args.weight,
                self.args.window,
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_elec,
            'ETTh2':Dataset_elec,
            'ETTm1':Dataset_elec,
            'ETTm2':Dataset_elec,
            'WTH':Dataset_elec,
            'TRAF':Dataset_elec,
            'EXCH':Dataset_elec,
            'ECL':Dataset_elec,
            'elec':Dataset_elec,
            'air':Dataset_airdelay,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            #Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            data_set = args.data,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        
        for i in range(5):
            print(data_set[0][i].shape, end=' ')
            print(' ')
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def loss_fn(self, mu, sigma ,labels):
        #zero_index = (labels != 0)
        distribution = torch.distributions.normal.Normal(mu, sigma)
        likelihood = distribution.log_prob(labels)
        return -torch.mean(likelihood)
        
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(vali_loader):
                pred, true, sample = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
                loss = criterion(sample, true)
                total_loss.append(loss.item())
            total_loss = np.average(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(train_loader):
                #print(batch_x.shape)
                iter_count += 1
                model_optim.zero_grad()
                true, sample = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in)
                loss = criterion(sample, true)
                train_loss.append(loss.item())
                #print(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            updatehidden = early_stopping.updatehidden
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(test_loader):
            true, sample = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
            preds.append(sample.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        preds = test_data.inverse_transform(preds)
        trues = test_data.inverse_transform(trues)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return mae, mse

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        preds = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(pred_loader):
            true, sample= self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = pred_data.inverse_transform(preds)
        preds = preds[...,-1].reshape(-1, preds.shape[-2])
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.savetxt(folder_path+'real_prediction.txt', preds, fmt='%.4f')
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                sample = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            sample= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            sample = dataset_object.inverse_transform(sample)
        
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        sample = sample[...,:,f_dim:]
        return batch_y, sample
#!/usr/bin/env python
# -*-encoding=utf8 -*-
import argparse
import os
import torch
import numpy as np
import random
from exp.exp_Model import Exp_Model

fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Very Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='Model',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, default='elec', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='elec2.csv', help='data file') #改文件名
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=192, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
parser.add_argument('--step_len', type=int, default=25, help='step length')

parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
parser.add_argument('--c_out', type=int, default=21, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--normal_layers', type=int, default=2, help='num of normal layers')
parser.add_argument('--enc_lstm', type=int, default=1, help='num of encoder lstm layers')
parser.add_argument('--dec_lstm', type=int, default=2, help='num of decoder lstm layers')
parser.add_argument('--weight', type=float, default=0.2, help='num of decoder lstm layers')
parser.add_argument('--window', type=float, default=2, help='size of sliding window')

parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=False)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='long', help='attention used in encoder, options:[long, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')

parser.add_argument('--do_test', type =bool, default=True, help='whether to produce test data of validation')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

print("Args: ")
print(args)
print('dimension')
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'freq':'h'},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'freq':'h'},
    'ETTm1':{'data':'ETTm1.csv', 'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'freq':'t'},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'freq':'t'},
    'WTH':{'data':'weather.csv','T':'T (degC)','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1], 'freq':'h'},
    'ECL':{'data':'electricity.csv','T':'MT_321','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1], 'freq':'t'},
    'TRAF':{'data':'traffic.csv','T':'tensor862','M':[861,861,861],'S':[1,1,1],'MS':[861,861,1],'freq':'h'},
    'EXCH':{'data':'exchange_rate.csv','T':'county8','M':[8,8,8],'S':[1,1,1],'MS':[8,8,1]},
    'elec':{'T':'target','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'freq':'t'},
    'air':{'T':'target','M':[6,6,6],'S':[1,1,1],'MS':[7,7,1], 'freq':'t'},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)
Exp = Exp_Model

pred_length = [48, 96, 192, 384, 768, 1536]
for i in range(len(pred_length)):
    all_mse = []
    all_mae = [] 
    args.pred_len = pred_length[i]
    for ii in range(args.itr):
    # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_step{}_dm{}_nh{}_el{}_dl{}_normal{}_elstm{}_dlstm{}_weight{}_window{}_df{}_at{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data_path, args.features, 
                args.seq_len, args.label_len, args.pred_len,args.step_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.normal_layers, args.enc_lstm, args.dec_lstm, args.weight, args.window, args.d_ff, args.attn,
                args.embed, args.distil, args.mix, args.des, ii)

        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
    
        if args.do_test:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse = exp.test(setting)
            all_mae.append(mae)
            all_mse.append(mse)
    #if args.do_predict:
    #    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #    exp.predict(setting, True)
        torch.cuda.empty_cache()
    print(np.mean(np.array(all_mse)), np.std(np.array(all_mse)), 
          np.mean(np.array(all_mae)), np.std(np.array(all_mae)))
