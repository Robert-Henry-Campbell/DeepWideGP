# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import os
import torch
# PyTorch 2.6+ defaults to weights_only=True, but PyTorch Lightning checkpoints
# contain argparse.Namespace and numpy objects that fail safe unpickling.
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
import random
import csv
import json
import datetime
import pickle
import numpy as np
import pytorch_lightning as pl


from sklearn import metrics
from models import DeepGP
from utils import SNPPCACHRDataModule
from genostore.datamodule import SNPStoreDataModule, is_store_dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from args_generator import args_initial 



base_folder = ''

model_dict = {'DeepGP':DeepGP }

# define the fieldnames of results to record
fieldnames = ['pheno','accuracy','precision','recall', 'specificity','f1','auc','mcc']

#%% set parameters
configs = args_initial()
fix_seed = configs.seed          # --seed (default 33, the published value; the study uses 1)
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if configs.data_dir is None:     # as published: --data_dir is honoured
    configs.data_dir = base_folder+'pukb/genes'
print('Running experiment for the whole genome')

#%% construct dataloaders
print("Loading data...")
# A dataset directory holding store.json is read lazily from the disk-backed genotype store
# (genostore/README.md); anything else goes through the original in-RAM pickle loader.
snpdata = SNPStoreDataModule(configs) if is_store_dataset(configs) else SNPPCACHRDataModule(configs)
configs.enc_len = len(snpdata.genes)
configs.enc_len_chr = [len(snp_chr) for snp_chr in snpdata.genes_chr]
configs.pos = snpdata.pos_chr
if configs.balanced_batches:     # half cases per batch; an epoch shows every non-case once (store datasets)
    from widedeepgp.samplers import balanced_one_disease
    snpdata.train_sampler = balanced_one_disease(snpdata.splits["train"]["y"], configs.batch_size, fix_seed)


def with_full_val_auroc(base):
    """The model, with validation AUROC over the whole validation set (val_auroc): the code's val_auc averages
    per-batch AUROCs, which a validation set at population prevalence (few or no cases per batch) distorts."""
    class Model(base):
        def on_validation_epoch_start(self):
            self._val = []

        def validation_step(self, batch, batch_idx):
            x_all, y = batch
            p = self(x_all)
            self.log('val_loss', self.criterion(p, y), sync_dist=True)
            self._val.append((p[:, 0].detach().float().cpu(), y[:, 0].cpu()))

        def on_validation_epoch_end(self):
            p, y = (torch.cat(t).numpy() for t in zip(*self._val))
            auc = metrics.roc_auc_score(y, p) if 0 < y.sum() < len(y) else float('nan')   # the sanity check sees 2 batches
            self.log('val_auroc', auc)
            print(f"validation at step {self.global_step}: val_auroc {auc:.4f}", flush=True)
    return Model


#%% build a model
if configs.dm == 'snps':
    model = with_full_val_auroc(model_dict[configs.mn].SNP_Model_mamba)(configs)
elif configs.dm == 'snps_covs': 
    model = with_full_val_auroc(model_dict[configs.mn].SNPCOV_Model_mamba)(configs)

monitor_acc = 'val_auroc'

if configs.save_log:
    log_name = f'{configs.mn}_{configs.dm}/{configs.label}'
    if len(configs.use_sim)>0:
         log_name = log_name+'/sim'
    logger = TensorBoardLogger(save_dir=base_folder+'DeepGP/logs/', name=log_name,version=configs.exp_name) #log_graph = True
    checkpoint_callback = ModelCheckpoint(monitor=monitor_acc,save_top_k=1,mode='max')
    callbacks = [EarlyStopping(monitor=monitor_acc, patience=configs.patience,mode="max"), checkpoint_callback]
    ecb = True
else:
    logger,ecb = False, False
    callbacks = [EarlyStopping(monitor=monitor_acc, patience=configs.patience,mode="max")]
    
trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=configs.gpus,
                    callbacks=callbacks,
                    max_epochs=30, 
                    val_check_interval=300,
                    check_val_every_n_epoch=None,   # every 300 batches, counted across epochs (shorter ones too)
                    logger=logger,
                    enable_checkpointing = ecb,
                    enable_progress_bar=False,
                    ) 

#%% start training
print("Start training...")

trainer.fit(model,snpdata) 

# testing
if configs.save_log:
        best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
else:
    best_model = model

# The test set is locked until every analysis is done (CLAUDE.md section 3), so the published test scoring
# is commented out below; the best model's validation-set predictions and a summary are written instead.
# trainer.test(best_model,snpdata) 
#
# #%% evaluation
# outputs = trainer.predict(best_model,snpdata)
# y_pred_proba = np.concatenate([opt[0].numpy() for opt in outputs])
# y_pred = np.concatenate([opt[1].numpy() for opt in outputs])
# y_true = np.concatenate([opt[2].numpy() for opt in outputs])
#
# # metrcis
# accuracy = metrics.accuracy_score(y_true, y_pred)
# precision = metrics.precision_score(y_true, y_pred)
# recall = metrics.recall_score(y_true, y_pred)
# # Specificity = TN / (TN + FP), where TN = true negatives, FP = false positives
# tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
# specificity = tn / (tn + fp)
# f1 = metrics.f1_score(y_true, y_pred)
# mcc = metrics.matthews_corrcoef(y_true, y_pred)
#
# auc = metrics.roc_auc_score(y_true, y_pred_proba)
# results_dict = {'pheno': configs.label,'accuracy':accuracy,'precision':precision,
#                 'recall':recall, 'specificity':specificity,'f1':f1,'auc':auc,'mcc':mcc}
#
#
# # open a csv to write results
# now = datetime.datetime.now().strftime('%m-%d_%H-%M')
# if configs.save_results:
#     results_dir = base_folder+f'DeepGP/results/{configs.mn}_{configs.dm}/{configs.label}/{configs.exp_name}'
#     if len(configs.use_sim)>0:
#         results_dir = base_folder+f'DeepGP/results/{configs.mn}_{configs.dm}/{configs.label}/sim/{configs.exp_name}'
#
#     if not os.path.exists(results_dir):
#             os.makedirs(results_dir)
#     results_csv = f'{results_dir}/results_{now}.csv'
#     with open(results_csv, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#     # open a json to write parameters
#     with open(f'{results_dir}/params_{now}.txt', 'w') as f:
#         configs.pos = None
#         json.dump(configs.__dict__, f, indent=2)
#     with open(results_csv, 'a', newline='') as f:
#         # Create the writer
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writerow(results_dict)
#     with open(f'{results_dir}/preds.pkl', 'wb') as f:
#         pickle.dump((y_true, y_pred_proba),f)
#
# if configs.show:
#     print(f'Accuracy: {accuracy}')
#     print(f'Precision: {precision}')
#     print(f'Recall: {recall}')
#     print(f'Specificity: {specificity}')
#     print(f'F1 Score: {f1}')
#     print(f'AUC: {auc}')
#     print(f'MCC: {mcc}')

results_dir = base_folder+f'DeepGP/results/{configs.mn}_{configs.dm}/{configs.label}/{configs.exp_name}'
os.makedirs(results_dir, exist_ok=True)
snpdata.predict_dataloader = snpdata.val_dataloader      # the validation set, never the locked test set
outputs = trainer.predict(best_model, datamodule=snpdata)   # through the data module, which decodes store batches
y_pred_proba = np.concatenate([opt[0].numpy() for opt in outputs])
y_true = np.concatenate([opt[2].numpy() for opt in outputs])
auc = metrics.roc_auc_score(y_true, y_pred_proba)
with open(f'{results_dir}/preds_val.pkl', 'wb') as f:
    pickle.dump((y_true, y_pred_proba), f)
y_train = np.asarray(snpdata.splits["train"]["y"])
summary = {'label': configs.label, 'rd_mode': configs.rd_mode, 'seed': fix_seed, 'val_auroc': float(auc),
           'best_val_auroc_logged': float(checkpoint_callback.best_model_score) if configs.save_log else None,
           'best_checkpoint': checkpoint_callback.best_model_path if configs.save_log else None,
           'steps': int(trainer.global_step), 'epochs': int(trainer.current_epoch),
           'snps': int(configs.enc_len), 'snps_per_chrom': [int(n) for n in configs.enc_len_chr],
           'train_people': int(len(y_train)), 'train_cases': int(np.nansum(y_train)), 'val_people': int(len(y_true))}
with open(f'{results_dir}/results.json', 'w') as f:
    json.dump(summary, f, indent=1)
print(f'AUC (validation set): {auc}')
