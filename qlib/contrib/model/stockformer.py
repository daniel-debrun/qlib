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

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def init_weights(module):
    """Randomly initialize weights for neural network modules."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Parameter):
        nn.init.normal_(module, mean=0.0, std=0.02)

class DecouplingFlowLayer(nn.Module):
    """
    Wavelet-based decoupling layer that decomposes only the stock return series into low and high frequency components.
    """
    def __init__(self, wavelet='haar', d_model=64, num_features=362, return_index=0):
        super(DecouplingFlowLayer, self).__init__()
        self.wavelet = wavelet
        self.d_model = d_model
        self.num_features = num_features
        self.return_index = return_index  # Index of the return feature (default 0)
        # Linear layers to project to d_model dimensions
        self.Wg = nn.Linear(num_features, d_model)
        self.Wh = nn.Linear(num_features, d_model)
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len, num_stocks, num_features)
        batch_size, seq_len, num_stocks, num_features = x.shape

        # Extract return series only (shape: batch_size, seq_len, num_stocks)
        return_series = x[:, :, :, self.return_index]

        # Reshape to (batch_size * num_stocks, seq_len) for DWT
        returns = return_series.permute(0, 2, 1).reshape(-1, seq_len)

        # Pad if necessary to make even-length
        if seq_len % 2 != 0:
            returns = F.pad(returns, (0, 1))

        # Apply single-level DWT to each (1D signal)
        Xl_raw, Xh_raw = [], []
        for i in range(returns.shape[0]):
            cA, cD = pywt.dwt(returns[i].cpu().numpy(), self.wavelet)
            Xl_raw.append(torch.tensor(cA, dtype=x.dtype, device=x.device))
            Xh_raw.append(torch.tensor(cD, dtype=x.dtype, device=x.device))

        Xl_raw = torch.stack(Xl_raw)
        Xh_raw = torch.stack(Xh_raw)

        # Interpolate back to original seq_len
        Xl_interp = F.interpolate(Xl_raw.unsqueeze(1), size=seq_len, mode='linear', align_corners=False).squeeze(1)
        Xh_interp = F.interpolate(Xh_raw.unsqueeze(1), size=seq_len, mode='linear', align_corners=False).squeeze(1)

        # Reshape to (batch_size, seq_len, num_stocks)
        Xl = Xl_interp.view(batch_size, num_stocks, seq_len).permute(0, 2, 1)
        Xh = Xh_interp.view(batch_size, num_stocks, seq_len).permute(0, 2, 1)

        others = x[:, :, :, 1:]

        F_prime = num_features - 1

        # Concatenate others + wavelet component
        Xl_cat = torch.cat([others, Xl.unsqueeze(-1)], dim=-1)  # (batch_size, seq_len, num_stocks, F_prime + 1)
        Xh_cat = torch.cat([others, Xh.unsqueeze(-1)], dim=-1)  # (batch_size, seq_len, num_stocks, F_prime + 1)

        # Linear projection to model dimension
        Xl_proj = self.Wg(Xl_cat.view(-1, F_prime + 1)).view(batch_size, seq_len, num_stocks, self.d_model)
        Xh_proj = self.Wh(Xh_cat.view(-1, F_prime + 1)).view(batch_size, seq_len, num_stocks, self.d_model)

        return Xl_proj, Xh_proj  # Both: (batch_size, seq_len, num_stocks, d_model)


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        # Layer normalization and feed-forward network
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x):
        # x: (batch_size, seq_len, num_stocks, d_model)
        batch_size, seq_len, num_stocks, d_model = x.shape
        
        # Flatten for multi-head attention
        x_flat = x.reshape(batch_size * seq_len, num_stocks, d_model)
        
        # Manage multi-head attention (Attention is all you need, Vaswani et al.)
        Q = self.query(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation (Attention is all you need, Vaswani et al.)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, num_stocks, d_model)
        output = self.out(attn_output)
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, num_stocks, d_model)
        
        # Residual connection and layer norm
        output = self.ln(output + x)
        
        # Feed forward
        output = self.ff(output) + output
        
        return output

class DualFrequencyEncoder(nn.Module):
    def __init__(self, seq_len, d_model, num_stocks, num_heads, kernel_size=2):
        super(DualFrequencyEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_stocks = num_stocks

        # Temporal attention for low-frequency components
        self.temporal_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # Dilated convolution with padding to preserve sequence length
        self.dilated_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, dilation=2, padding=1)
        self.relu = nn.ReLU()

        # Positional encoding
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        # Time embedding
        self.num_time_slots = 252
        self.time_embedding = nn.Linear(self.num_time_slots, d_model)

        # Spatial embedding
        self.spatial_embedding = nn.Parameter(torch.randn(num_stocks, d_model))

        # GraphAttentionLayer 
        self.gat_l = GraphAttentionLayer(d_model, num_heads=num_heads)
        self.gat_h = GraphAttentionLayer(d_model, num_heads=num_heads)
        
        # Initialize weights
        self.apply(init_weights)
        # Initialize spatial embedding parameter specifically
        nn.init.normal_(self.spatial_embedding, mean=0.0, std=0.02)

    def embed_time_slots(self, ts_index):
        # ts_index: (batch_size, seq_len), values in [0, 251]
        ts_index = ts_index.long()  # Ensure integer type for one_hot
        one_hot = F.one_hot(ts_index, num_classes=self.num_time_slots).float()  # (batch, seq, 252)
        time_embed = self.time_embedding(one_hot)  # (batch, seq, d_model)
        return time_embed  # (batch, seq, d_model)

    def embed_spatial(self, returns):
        # returns: (batch_size, seq_len, num_stocks)
        batch_size, seq_len, num_stocks = returns.shape
        
        # Compute correlation for each sample in the batch on GPU
        spatial_embeddings = []
        for b in range(batch_size):
            corr_matrix = self.compute_correlation_gpu(returns[b])  # Keep on GPU
            spatial_emb = torch.matmul(corr_matrix, self.spatial_embedding)
            spatial_embeddings.append(spatial_emb)
        
        spatial_embeddings = torch.stack(spatial_embeddings, dim=0)
        spatial_embeddings = spatial_embeddings.unsqueeze(1).repeat(1, seq_len, 1, 1)
        
        return spatial_embeddings
    
    def embed_spatial_batched(self, returns):
        # returns: (batch_size, seq_len, num_stocks)
        batch_size, seq_len, num_stocks = returns.shape
        
        # Compute correlation for entire batch at once
        returns_centered = returns - returns.mean(dim=1, keepdim=True)
        returns_std = returns_centered.std(dim=1, keepdim=True) + 1e-8
        returns_normalized = returns_centered / returns_std
        
        # Batch matrix multiplication for correlation
        # returns_normalized: (batch_size, seq_len, num_stocks)
        # Transpose for batch correlation: (batch_size, num_stocks, seq_len)
        returns_T = returns_normalized.transpose(1, 2)
        # Batch correlation: (batch_size, num_stocks, num_stocks)
        corr_matrices = torch.bmm(returns_T, returns_normalized) / (seq_len - 1)
        
        # Clamp and set diagonal
        corr_matrices = torch.clamp(corr_matrices, -1.0, 1.0)
        eye = torch.eye(num_stocks, device=returns.device).unsqueeze(0).expand(batch_size, -1, -1)
        corr_matrices = corr_matrices * (1 - eye) + eye
        
        # Apply spatial embedding: (batch_size, num_stocks, d_model)
        spatial_embeddings = torch.bmm(corr_matrices, self.spatial_embedding.unsqueeze(0).expand(batch_size, -1, -1))
        spatial_embeddings = spatial_embeddings.unsqueeze(1).repeat(1, seq_len, 1, 1)
        
        return spatial_embeddings
    
    def compute_spearman_correlation(self, returns):
        ## returns: (seq_len, num_stocks)
        seq_len, num_stocks = returns.shape
        
        # Standardize the returns on GPU
        returns_centered = returns - returns.mean(dim=0, keepdim=True)
        returns_std = returns_centered.std(dim=0, keepdim=True) + 1e-8
        returns_normalized = returns_centered / returns_std
        
        # Compute correlation matrix on GPU
        corr_matrix = torch.mm(returns_normalized.T, returns_normalized) / (seq_len - 1)
        
        # Clamp to valid correlation range
        corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
        
        # Fill diagonal with 1s
        corr_matrix.fill_diagonal_(1.0)
        
        return corr_matrix
       

    def forward(self, X_l, X_h, ts_index, returns):
        # X_l, X_h: (batch_size, seq_len, num_stocks, d_model)
        # ts_index: (batch_size, seq_len)
        # returns: (batch_size, seq_len, num_stocks)
        batch_size, seq_len, num_stocks, d_model = X_l.shape

        # Temporal Attention on low-frequency
        X_l_flat = X_l.reshape(batch_size * num_stocks, seq_len, d_model)
        X_tatt, _ = self.temporal_attention(X_l_flat, X_l_flat, X_l_flat)  # (batch_size*num_stocks, seq_len, d_model)
        X_tatt = X_tatt.reshape(batch_size, seq_len, num_stocks, d_model)

        # Dilated Conv on high-frequency
        X_h_perm = X_h.permute(0, 2, 3, 1)  # (batch_size, num_stocks, d_model, seq_len)
        X_h_flat = X_h_perm.reshape(batch_size * num_stocks, d_model, seq_len)
        X_conv = self.dilated_conv(X_h_flat)  # (batch_size*num_stocks, d_model, seq_len)
        X_conv = self.relu(X_conv)
        X_conv = X_conv.reshape(batch_size, num_stocks, d_model, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, d_model)
        
        # Temporal Graph Embedding (252 time slots [trading days])
        p_tem = self.embed_time_slots(ts_index)  # (batch, seq, d_model)
        p_tem = p_tem.unsqueeze(2).repeat(1, 1, num_stocks, 1)  # (batch, seq, num_stocks, d_model)

        # Spatial Embedding
        p_spa = self.embed_spatial_batched(returns)  # GPU-optimized version
        
        # Concat Embeddings
        
        X_l = X_tatt + p_tem + p_spa
        X_h = X_conv + p_tem + p_spa

        # Self-Attention
        X_l_gat = self.gat_l(X_l)  # (batch, seq, num_stocks, d_model)
        X_h_gat = self.gat_h(X_h)  # (batch, seq, num_stocks, d_model)

        return X_l_gat, X_h_gat

class DualFrequencyFusionDecoder(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, num_heads):
        super(DualFrequencyFusionDecoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Predictors
        self.predictor_l = nn.Conv2d(self.seq_len, self.pred_len, (1,1))
        self.predictor_h = nn.Conv2d(self.seq_len, self.pred_len, (1,1))

        # Positional encoding (Attention is all you need, Vaswani et al.)
        position = torch.arange(pred_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(pred_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        # Multi-attention layers
        self.attn_self = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

        # Output layers
        self.fc_reg= nn.Linear(d_model, 1)
        self.fc_cla = nn.Linear(d_model, 2) # (binary classification)
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, X_l_gat, X_h_gat) -> dict:
        # X_l_gat, X_h_gat: (batch_size, seq_len, num_stocks, d_model)
        
        # Predictors
        pred_l = self.predictor_l(X_l_gat)  # (batch_size, pred_len, num_stocks, d_model)
        pred_h = self.predictor_h(X_h_gat)  # (batch_size, pred_len, num_stocks, d_model)

        batch_size, pred_len, num_stocks, d_model = pred_l.shape

        pred_l = pred_l.reshape(batch_size * num_stocks, pred_len, d_model)
        pred_h = pred_h.reshape(batch_size * num_stocks, pred_len, d_model)

        # Add positional encoding
        input_l = pred_l + self.positional_encoding  # (batch_size*num_stocks, pred_len, d_model)
        input_h = pred_h + self.positional_encoding  # (batch_size*num_stocks, pred_len, d_model)

        # Fusion Attention
        attn_self_out, _ = self.attn_self(input_l, input_l, input_l)
        attn_cross_out, _ = self.attn_cross(input_l, input_h, input_h)
        output = attn_self_out + attn_cross_out  # (batch_size*num_stocks, pred_len, d_model)

        # Low-frequency outputs
        Y_l_reg = self.fc_reg(pred_l).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)
        Y_l_cla = self.fc_cla(pred_l).reshape(batch_size, pred_len, num_stocks, 2)  # (batch_size, pred_len, num_stocks)
        
        # Fused outputs
        Y_reg = self.fc_reg(output).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)
        Y_cla = self.fc_cla(output).reshape(batch_size, pred_len, num_stocks, 2)  # (batch_size, pred_len, num_stocks)
        
        return {
            "lreg": Y_l_reg, # Low-frequency regression output
            "lcla": Y_l_cla, # Low-frequency classification output
            "reg": Y_reg, # Raw regression output
            "cla": Y_cla, # Raw classification output
        }

class StockFormer(nn.Module):
    """
    StockFormer model for time series forecasting.
    """
    def __init__(self, num_stocks, seq_len=20, pred_len=2, num_features=362, d_model=128, num_heads=2, dropout=0.2, pred_features=[0, 1], **kwargs):
        super(StockFormer, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.pred_features = pred_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.dropout = dropout

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Wavelet-based decoupling layer
        self.decouple = DecouplingFlowLayer(d_model=d_model, num_features=num_features)

        # Dual-frequency spatiotemporal encoder
        self.dual_freq_encoder = DualFrequencyEncoder(seq_len=seq_len, d_model=d_model, num_stocks=num_stocks, num_heads=num_heads)

        # Dual-frequency fusion decoder
        self.dual_freq_fusion_decoder = DualFrequencyFusionDecoder(seq_len=seq_len, pred_len=pred_len, d_model=d_model, num_heads=num_heads)
        
        # Initialize weights for the entire model
        self.apply(init_weights)

        self.fitted = False
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")


    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")


    def forward(self, x, ts) -> dict:
        # x: (batch_size, seq_len, num_stocks, num_features)
        # ts: (batch_size, seq_len) time slot indices
        
        # Input validation
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {x.shape[1]}")
        if x.shape[3] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[3]}")
        if ts.shape[1] != self.seq_len:
            raise ValueError(f"Time indices shape {ts.shape} doesn't match sequence length {self.seq_len}")
        
        # Decompose input into low and high frequency components
        X_l, X_h = self.decouple(x)  # Both: (batch_size, seq_len, num_stocks, d_model)
        
        # Apply dropout to the decomposed features
        X_l = self.dropout(X_l)
        X_h = self.dropout(X_h)

        returns = x[..., self.pred_features[0]]  # (batch_size, seq_len, num_stocks)

        # Encode
        X_l_gat, X_h_gat = self.dual_freq_encoder(X_l, X_h, ts, returns)

        # Apply dropout to encoded features
        X_l_gat = self.dropout(X_l_gat)
        X_h_gat = self.dropout(X_h_gat)

        # Decode
        predictions = self.dual_freq_fusion_decoder(X_l_gat, X_h_gat)

        return predictions
    
    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())
    
def output_to_raw_numpy(out, lookahead=0):
    """
    Convert model output dictionary to numpy arrays for further analysis or signal generation.
    Args:
        out (dict): Model output dictionary with keys 'reg', 'lreg', 'cla', 'lcla'.
        lookahead (int): Index for prediction horizon (default 0).
    Returns:
        reg_pred: Regression predictions (raw output).
        reg_l_pred: Low-frequency regression predictions.
        cla_pred: Classification predictions (argmax over classes).
        cla_l_pred: Low-frequency classification predictions.
        cla_probs: Classification probabilities (softmaxed).
        cla_l_probs: Low-frequency classification probabilities.
    """
    reg_pred = out["reg"][:, lookahead, :].cpu().numpy()
    reg_l_pred = out["lreg"][:, lookahead, :].cpu().numpy()

    cla_pred = torch.argmax(out["cla"][:, lookahead, :], dim=-1).cpu().numpy()
    cla_l_pred = torch.argmax(out["lcla"][:, lookahead, :], dim=-1).cpu().numpy()

    cla_probs = torch.softmax(out["cla"][:, lookahead, :], dim=-1).cpu().numpy()
    cla_l_probs = torch.softmax(out["lcla"][:, lookahead, :], dim=-1).cpu().numpy()

    return reg_pred, reg_l_pred, cla_pred, cla_l_pred, cla_probs, cla_l_probs

def output_to_signals(out, lookahead=0):
    """
    Generate trading signals (-1, 0, 1) based on model predictions and classification probabilities.
    Args:
        out (dict): Model output dictionary.
        lookahead (int): Index for prediction horizon (default 0).
    Returns:
        signals: Array of trading signals for each stock and time step.
    """
    reg_pred, reg_l_pred, cla_pred, cla_l_pred, cla_probs, cla_l_probs = output_to_raw_numpy(out, lookahead)
    num_Stocks = reg_pred.shape[1]
    signals = np.zeros_like(reg_pred)
    for t in range(len(reg_pred)):
        for stock in range(num_Stocks):
            # If probability of class 1 (up) > 0.5, use regression sign for buy/sell
            if cla_probs[t, stock, 1] > 0.5:
                if reg_pred[t, stock] > 0:
                    signals[t, stock] = 1.0
                else:
                    signals[t, stock] = -1.0
            else:
                # Otherwise, use classification prediction for buy/hold
                if cla_pred[t, stock] > 0:
                    signals[t, stock] = 1.0
                else:
                    signals[t, stock] = 0.0
    return signals

def create_compiled_stockformer(device='cuda', **kwargs):
    """
    Create and compile a StockFormer model instance with configurable parameters.
    Args:
        device (str): Device to run the model on ('cuda' or 'cpu').
        **kwargs: Additional parameters for StockFormer initialization.
    Returns:
        Compiled StockFormer model instance.
    """
    #torch.set_float32_matmul_precision("high")
    model = StockFormer(**kwargs)
    model.to(device)
    
    # Only use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")
    
    # Compile the model for better performance
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled successfully")
    except Exception as e:
        print(f"Model compilation failed: {e}, using non-compiled version")
    
    return model
