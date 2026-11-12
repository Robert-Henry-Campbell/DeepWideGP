# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import argparse
PHENO_CONT = False
def args_initial(argv=None):
    parser = argparse.ArgumentParser()
    # data define
    parser.add_argument('--SNP_len', type=int, default=50, help='length of SNP windows')
    parser.add_argument('--data_dir', type=str, default=None, help='directory of datasets')

    # model define
    parser.add_argument('--mn', type=str, default='DeepGP', help='model name')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input dimension')
    parser.add_argument('--cov_dim', type=int, default=8, help='covariate dimension')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=1, help='num of SNP encoder layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in encoder')
    parser.add_argument('--snp_embed', type=str, default='cov', help='embedding method: cov or vocab')
    parser.add_argument('--final_pool', type=str, default='concat', help='output head type: last, concat, atten')
    parser.add_argument('--use_pcs', action='store_false',default=True, help='whether to use population structure')
    parser.add_argument('--bi_direct', action='store_true', default=False, help='whether to bi-directional Mamba')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout to avoid overfitting')
    parser.add_argument('--fused_add_norm', action='store_false', default=True, help='whether to use residuals')
    parser.add_argument('--n_heads',type=int, default=8, help='number of heads in MHSA')
    parser.add_argument('--n_fcn',type=int, default=2, help='number of hidden layers in FCN')
    parser.add_argument('--l2',type=float, default=1e-5, help='weight decay (L2) in Adam')

    # training define
    parser.add_argument('--gpus', type=int, default=1, help='gpu devices')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--keep', action='store_false', default=True, help='keep saved model if it exists')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--loss_name',  default=None, help='name of loss function')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--threshold', type=int, default=0.5, help='threshold for transforming probability to binary')  


    # display
    parser.add_argument('--show', action='store_false', default=True, help='display metrics')

    # experiment
    parser.add_argument('--exp_name', default=None, help='name of experiment')
    parser.add_argument('--save_log', default=True, help='save logs for experiment')
    parser.add_argument('--save_results', default=True, help='save results experiment')
    parser.add_argument('--label', default=None, help='target phenotype of experiment')
    parser.add_argument('--dm', default='snps_covs', help='data mode: snps or snps_covs')
    parser.add_argument('--ve', default=None, help='Name of external validation cohort')
    parser.add_argument('--rd_mode', default='ld_all', help='read data mode')
    parser.add_argument('--patience',type=int, default=10, help='patience number in early stopping')
    parser.add_argument('--use_sim', type=str, default='', help='whether to use simulated phenotype')
    parser.add_argument('--use_cov', action='store_false', default=True, help='whether to use covariate data')
    parser.add_argument('--scaling', action='store_false', default=True, help='whether to scale data')
    parser.add_argument('--non_add', action='store_true', default=False, help='whether to use non-additive genes only')
    parser.add_argument('--shuffle', action='store_true', default=False, help='whether to randomly shuffle data')
    parser.add_argument('--cont_pheno', action='store_true', default=False, help='whether the phenotype is continuous')
    # genotype-store datasets only (ignored by the pickle loader)
    parser.add_argument('--snp_stats', type=str, default='store', help="SNP scaling statistics: 'store' (computed over all samples at conversion) or 'train' (recomputed on the training split, as the pickle loader does)")
    parser.add_argument('--pos_mode', type=str, default=None, help="positions for the positional embedding: 'index' or 'pos' (default: what store.json says)")

    # ---- multi-disease training (widedeepgp / main_multilabel.py; main_genome.py ignores these) ----
    parser.add_argument('--labels', type=str, default=None, help='comma-separated subset of the dataset\'s label columns (default: all of them)')
    parser.add_argument('--mixer', type=str, default='bimamba1', help="sequence block: 'bimamba1' (the published block) or 'mamba3'")
    parser.add_argument('--sampler', type=str, default='rotation', help="'rotation' (each batch balanced on one disease) or 'shuffled'")
    parser.add_argument('--loss_outputs', type=str, default='all', help="'all': every output learns from every batch; 'target': only the disease the batch was balanced on")
    parser.add_argument('--gamma', type=float, default=0.5, help='class-weight exponent: 0 none, 0.5 square root, 1 full balance')
    parser.add_argument('--w_max', type=float, default=20.0, help='cap on the class weight of a case')
    parser.add_argument('--no_sampling_correction', action='store_true', default=False, help='do not divide each person\'s loss by their sampling rate')
    parser.add_argument('--head_hidden', type=int, default=256, help='width of the shared hidden layer before the outputs')
    parser.add_argument('--annot', action='store_true', default=False, help='use per-SNP annotation vectors (the dataset must provide them)')
    parser.add_argument('--precision', type=str, default='32', help="'32' or 'bf16' (bf16 autocast with pooling, head and loss in float32)")
    parser.add_argument('--grad_ckpt', action='store_true', default=False, help='activation checkpointing per input sequence')
    parser.add_argument('--accumulate', type=int, default=1, help='batches accumulated per optimiser step')
    parser.add_argument('--val_per_epoch', type=float, default=4.0, help='validation passes per epoch-equivalent')
    parser.add_argument('--max_epoch_equiv', type=float, default=30.0, help='stop after this many epoch-equivalents')
    parser.add_argument('--stop_patience', type=int, default=8, help='validation passes without improvement before stopping (0: never stop early)')
    parser.add_argument('--warmup_steps', type=int, default=0, help='linear learning-rate warm-up, in optimiser steps')
    parser.add_argument('--lr_schedule', type=str, default='constant', help="'constant' or 'cosine'")
    parser.add_argument('--seed', type=int, default=33, help='random seed')
    parser.add_argument('--run_dir', type=str, default='runs', help='where runs are written')
    parser.add_argument('--experiment', type=str, default='default', help='experiment name (a folder of runs)')
    parser.add_argument('--resume', action='store_true', default=False, help='continue the run of this name from its last checkpoint, on the same data with the same settings (it may train longer)')
    parser.add_argument('--init_from', type=str, default=None, help='snapshot whose weights initialise the model (all but the output layer if the labels differ)')
    parser.add_argument('--snapshots_per_decade', type=int, default=8, help='weight snapshots per tenfold increase in steps')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='batch size for validation and scoring (default: the training batch size)')
    parser.add_argument('--steps_per_epoch_equiv', type=int, default=None, help='override the number of batches in an epoch-equivalent (testing)')
    parser.add_argument('--m3_d_state', type=int, default=None, help='Mamba-3 state size (default: 16, the state size of the published block)')

    args = parser.parse_args(argv)
    return args