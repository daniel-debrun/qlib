# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Code based on StockFormer:

@article{MA2025126803,
title = {Stockformer: A price–volume factor stock selection model based on wavelet transform and multi-task self-attention networks},
journal = {Expert Systems with Applications},
volume = {273},
pages = {126803},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126803},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425004257},
author = {Bohan Ma and Yushan Xue and Yuan Lu and Jing Chen},
keywords = {Price–volume factor selection, Stockformer, Wavelet transform, Spatiotemporal graph embedding, TopK-Dropout strategy},
abstract = {As the Chinese stock market continues to evolve and its market structure grows increasingly complex, traditional quantitative trading methods face escalating challenges. Due to policy uncertainty and frequent market fluctuations triggered by sudden economic events, existing models often struggle to predict market dynamics accurately. To address these challenges, this paper introduces “Stockformer,” a price–volume factor stock selection model that integrates wavelet transformation and a multitask self-attention network to enhance responsiveness and predictive accuracy regarding market instabilities. Through discrete wavelet transform, Stockformer decomposes stock returns into high and low frequencies, meticulously capturing long-term market trends and short-term fluctuations, including abrupt events. Moreover, the model incorporates a Dual-Frequency Spatiotemporal Encoder and graph embedding techniques to capture complex temporal and spatial relationships among stocks effectively. Employing a multitask learning strategy, it simultaneously predicts stock returns and directional trends. Experimental results show that Stockformer outperforms existing advanced methods on multiple real stock market datasets. In strategy backtesting, Stockformer consistently demonstrates exceptional stability and reliability across market conditions—whether rising, falling, or fluctuating—particularly maintaining high performance during downturns or volatile periods, indicating high adaptability to market fluctuations. To foster innovation and collaboration in the financial analysis sector, the Stockformer model’s code has been open-sourced and is available on the GitHub repository: https://github.com/Eric991005/Multitask-Stockformer.}
}
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...model.base import Model
from qlib.workflow import R
import torch
import math

from multitask_stockformer import StockFormer as MultitaskStockFormer
from multitask_stockformer.lib.graph_utils import loadGraph, StockDataset, disentangle, generate_temporal_embeddings
from multitask_stockformer.ipynb_checkpoint.MultiTask_Stockformer_train_checkpoint import train, test


class StockDatasetQlib(StockDataset):
    def __init__(self, df, args, mode='train'):
        self.mode = mode
        # Load data
        # TODO: VERIFY THE DATA FORMAT 
        Traffic = df #TODO: Col corresponding to RETURN data
        indicator = df #TODO: Col corresponding to TREND_DIRECTION data
        concatenated_arr = df # Cols corresponding to ALPHA360
        bonus_all = concatenated_arr
        num_step = Traffic.shape[0]
        train_steps = round(args.train_ratio * num_step)
        test_steps = round(args.test_ratio * num_step)
        val_steps = num_step - train_steps - test_steps
        TE = generate_temporal_embeddings(num_step, args)
        if mode == 'train':
            data_slice = slice(None, train_steps)
        elif mode == 'val':
            data_slice = slice(train_steps, train_steps + val_steps)
        else:  # mode == 'test'
            data_slice = slice(-test_steps, None)
        self.data = Traffic[data_slice]
        self.indicator = indicator[data_slice]
        self.bonus_all = bonus_all[data_slice]
        self.TE = TE[data_slice]
        self.X, self.Y = self.seq2instance(self.data, args.T1, args.T2)
        self.XL, self.XH = disentangle(self.X, args.w, args.j)
        self.YL, self.YH = disentangle(self.Y, args.w, args.j)
        self.indicator_X, self.indicator_Y = self.seq2instance(self.indicator, args.T1, args.T2)
        self.bonus_X, self.bonus_Y = self.bonus_seq2instance(self.bonus_all, args.T1, args.T2)
        self.TE = self.seq2instance(self.TE, args.T1, args.T2)
        self.TE = np.concatenate(self.TE, axis=1).astype(np.int32)
        # Adding the infea attribute based on bonus_all
        self.infea = bonus_all.shape[-1] + 2  # Last dimension of bonus_all plus one

class StockFormer(Model):
    """StockFormer Model"""
# TODO: convert DatasetH into StockDataset expected format
# TODO: replace workflow config data files with qlib equivalent paths and modules
    def __init__(self, **kwargs):
        super(MultitaskStockFormer, self).__init__(**kwargs)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(
        self,
        dataset,
        args,
        evals_result=dict(),
    ):
        """Qlib calls the fit method to train the model"""
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        outfea_class = 2
        outfea_regress = 1
        # trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infeature = loadData(args)
        train_dataset = StockDataset(dl_train, args, mode='train')
        val_dataset = StockDataset(dl_valid, args, mode='val')
        # get data
        # train data
        trainXL = train_dataset.XL
        trainXH = train_dataset.XH
        trainXC = train_dataset.indicator_X
        trainTE = train_dataset.TE
        trainY = train_dataset.Y
        trainYL = train_dataset.YL
        trainYC = train_dataset.indicator_Y
        bonus_trainX = train_dataset.bonus_X
        # val data
        valXL = val_dataset.XL
        valXH = val_dataset.XH
        valXC = val_dataset.indicator_X
        valTE = val_dataset.TE
        valY = val_dataset.Y
        valYL = val_dataset.YL
        valYC = val_dataset.indicator_Y
        bonus_valX = val_dataset.bonus_X
        # infeature number
        infeature = train_dataset.infea
        # adj, graphwave = loadGraph(args)
        adjgat = loadGraph(args)
        self.adjgat = torch.from_numpy(adjgat).float().to(self.device)
        print("loading end....")

        print("constructing model begin....")
        self.model = MultitaskStockFormer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
        print("constructing model end....")

        print("training begin....")
        train(self.model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
        print("training end....")

    def load(self, args):
        self.model.load_state_dict(torch.load(args.model_file))

    def predict(self, dataset: DatasetH, args):
        if self.model is None:
            try:
                self.load(args)
            except Exception as e:
                raise ValueError(f"Failed to load model, please call fit if no model exists: {e}")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")

        test_dataset = StockDataset(dl_test, args, mode='test')
        testXL = test_dataset.XL
        testXH = test_dataset.XH
        testXC = test_dataset.indicator_X
        testTE = test_dataset.TE
        testY = test_dataset.Y
        testYL = test_dataset.YL
        testYC = test_dataset.indicator_Y
        bonus_testX = test_dataset.bonus_X
        print("testing begin....")
        test(self.model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, self.adjgat)
        self.model.eval()
        num_val = testXL.shape[0]
        num_batch = math.ceil(num_val / args.batch_size)

        pred_class = []
        pred_regress = []
        label_class = []
        label_regress = []

        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                    xl = torch.from_numpy(testXL[start_idx : end_idx]).float().to(self.device)
                    xh = torch.from_numpy(testXH[start_idx : end_idx]).float().to(self.device)
                    xc = torch.from_numpy(testXC[start_idx : end_idx]).float().to(self.device)
                    te = torch.from_numpy(testTE[start_idx : end_idx]).to(self.device)
                    bonus = torch.from_numpy(bonus_testX[start_idx : end_idx]).float().to(self.device)
                    y = testY[start_idx : end_idx]
                    yc = testYC[start_idx : end_idx]
                    

                    hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = self.model(xl, xh, te, bonus, xc, self.adjgat)

                    pred_class.append(hat_y_class.cpu().numpy())
                    pred_regress.append(hat_y_regress.cpu().numpy())
                    label_class.append(yc)
                    label_regress.append(y)

        pred_class = np.concatenate(pred_class, axis=0)
        pred_regress = np.concatenate(pred_regress, axis=0)
        label_class = np.concatenate(label_class, axis=0)
        label_regress = np.concatenate(label_regress, axis=0)
        print("testing end....")
        # TODO: output probability to choose regression or classification
        return pd.Series(pred_class, index= yc.index) # Higher accuracy from classification when confidence > 0.2 

    def finetune(self, dataset: DatasetH):
        pass

