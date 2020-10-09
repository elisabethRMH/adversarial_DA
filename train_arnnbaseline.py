#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:18:28 2020

@author: ehereman
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"

import logging
from datetime import datetime
import sys
from scipy.io import loadmat, savemat
import numpy as np
import tensorflow as tf

#from mean_teacher_E.model import ModelE
from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config

sys.path.insert(0, "/users/sista/ehereman/GitHub/mean-teacher/tensorflow")
#from experiments.run_context import RunContext
#from datasets import Cifar10ZCA
#from mean_teacher import minibatching

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_adversarialV import SubGenFromFile

filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"
#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
#filename='data_split_eval.mat' # For a while while debugging

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch4'; # no overlap
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap
#source='/users/sista/ehereman/Desktop/all_data_epoch4'

fold=0
print('Fold: ', fold)
train_files=files_folds['train_sub']#[fold][0][0]
eval_files=files_folds['eval_sub']#[fold][0][0]
test_files=files_folds['test_sub']#[fold][0][0]


training_epoch=85
training_length=100000
#percent_unlabeled=0.9

config= Config()
config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_4'
config.checkpoint_dir= './checkpoint/'
config.allow_soft_placement=True
config.log_device_placement=False
config.nchannel=1
config.domainclassifierstage2 = False
config.add_classifieroutput=False
# path where checkpoint models are stored

#logging.basicConfig(level=logging.INFO)
#LOG = logging.getLogger('main')

def run(percent_unlabeled, normalize=True):
#    config.out_path = os.path.join(config.out_dir, 'SEMISUP{}unlabeled'.format(percent_unlabeled))
#    #config.out_path = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
#    config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
#    if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
#    if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
#    config.domainclassifier = True
    
    train_generator= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=normalize) #TODO adapt back
    train_generator.switch_channels(channel_to_use=1)

    test_generator =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=normalize)
    test_generator.switch_channels(channel_to_use=1)
    
    #Eval set is normal eval set
    eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=normalize)
    eval_generator.switch_channels(channel_to_use=1)
#    train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
#    eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
#    test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
#    
#    #E: nb of epochs in each set (in the sense of little half second windows)
#    print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
#    
#    #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
#    print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
#
#    
#    with tf.Graph().as_default():
#        session_conf = tf.ConfigProto(
#          allow_soft_placement=config.allow_soft_placement,
#          log_device_placement=config.log_device_placement)
#        session_conf.gpu_options.allow_growth = True
#        sess = tf.Session(config=session_conf)
#        with sess.as_default():
#
#            model = AdversarialNetwork(config, session=sess)
#            model.initialize()
#    
#            model.train( train_generator, eval_generator, test_generator, training_epoch, training_length)
#    tf.reset_default_graph()
    #tf.clear_all_variables()
#    #####################################################################################

    #config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/try3/FULLYSUP{}unlabeled'.format(percent_unlabeled)
    config.out_path = os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(percent_unlabeled))
    #config.out_path = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
    config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
    if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
    if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
    config.domainclassifier = False

    #train_generator.prune_unlabeled()
    #train_generator._normalize() #New way of normalizing, after try4
    
    #E: nb of epochs in each set (in the sense of little half second windows)
    print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
    train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
    eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
    test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)    
    #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
    print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            model = AdversarialNetwork(config, session=sess)
            model.initialize()
            model.train( train_generator, eval_generator, test_generator, training_epoch, training_length)
    #model.test(test_generator)

if __name__ == "__main__":
    run(0.0)
    config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_5'
    run(0.0)
    config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_6'
    run(0.0)
    config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_7'
    run(0.0)
    

