#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:10:30 2020

@author: ehereman
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
#from subgenfromfile_loadrandomtuples import SubGenFromFile
#from subgenfromfile_epochsave import SubGenFromFile
from subgenfromfile_adversarialV import SubGenFromFile

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
#filename='data_split_eval.mat'
filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap

root = '/esat/biomeddata/ehereman/MASS_toolbox'

fold=0
print('Fold: ', fold)
train_files=files_folds['train_sub']#[fold][0][0]
eval_files=files_folds['eval_sub']#[fold][0][0]
test_files=files_folds['test_sub']#[fold][0][0]

# Parameters
# ==================================================
#E toevoeging
FLAGS = tf.app.flags.FLAGS
for attr, value in sorted(FLAGS.__flags.items()): # python3
    x= 'FLAGS.'+attr
    exec("del %s" % (x))
    

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
percent_unlabeled=0.0
tf.app.flags.DEFINE_string("out_dir", '/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_2/', "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")

tf.app.flags.DEFINE_integer("seq_len", 32, "Sequence length (default: 32)")

tf.app.flags.DEFINE_integer("nfilter", 20, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 32, "Sequence length (default: 20)")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.join(FLAGS.out_dir, 'FULLYSUP{}unlabeled'.format(percent_unlabeled))
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.attention_size1 = FLAGS.attention_size1
config.training_epoch = int(40) #/6 if using load_random_tuple
config.out_dir = FLAGS.out_dir
config.checkpoint_dir= './checkpoint/'
config.allow_soft_placement=True
config.log_device_placement=False
config.nchannel=1
config.out_path = out_path
config.checkpoint_path = checkpoint_path
config.domainclassifier=False

train_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=True) #TODO adapt back
eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)


train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)


print("Test set: {:d}".format(len(test_generator._indices)))

print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))



# variable to keep track of best fscore
best_fscore = 0.0
best_acc = 0.0
best_kappa = 0.0
min_loss = float("inf")
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        arnn=AdversarialNetwork(config, session=sess)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(arnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        #sess.run(tf.initialize_all_variables())
        best_dir = os.path.join(checkpoint_path, 'best_model_acc')#"best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")


        def dev_step(x_batch, y_batch):
            frame_seq_len = np.ones(len(x_batch),dtype=int) * config.frame_seq_len
            feed_dict = {
                arnn.input_x: x_batch,
                arnn.input_y: y_batch,
                arnn.dropout_keep_prob_rnn: 1.0,
                arnn.frame_seq_len: frame_seq_len
            }
            output_loss, domain_loss, total_loss, yhat, score, features = sess.run(
                   [arnn.output_loss_mean,arnn.domain_loss_mean, arnn.loss, arnn.predictionC, arnn.score_C, arnn.attention_out1], feed_dict)
            return output_loss, total_loss, yhat, score, features
        
        def evaluate(gen):
            # Validate the model on the entire evaluation test set after each epoch
            feat = np.zeros([len(gen.datalist),config.nhidden1*2])
            
            #yhat = np.zeros(len(gen.tuples))
            #score = np.zeros([len(gen.tuples), config.nclass])
            num_batch_per_epoch = len(gen)
            test_step = 0
            ygt = np.zeros(len(gen.datalist))
            while test_step < num_batch_per_epoch:
                (x_batch,y_batch)=gen[test_step]
                x_batch=x_batch[:,0,:,:,0:3]
                y_batch=y_batch[:,0]
                output_loss_, total_loss_, yhat_, score_, features = dev_step(x_batch, y_batch) #E: score is raw output without applying softmax?
                feat[(test_step)*config.batch_size : (test_step+1)*config.batch_size, :] = features
                ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                test_step+=1
            if len(gen.datalist) - test_step*config.batch_size==1:
                ygt=ygt[0:-1]
                feat=feat[0:-1]
            elif len(gen.datalist) > test_step*config.batch_size:
                (x_batch,y_batch)=gen[test_step]
                x_batch=x_batch[:,0,:,:,0:3]
                y_batch=y_batch[:,0]
                output_loss_, total_loss_, yhat_, score_, features = dev_step(x_batch, y_batch)
                ygt[ (test_step)*config.batch_size : len(gen.datalist)] = y_batch
                feat[test_step*config.batch_size:len(gen.datalist)] = features
            ygt+=1
            return ygt, feat
        

        ygt, feat = evaluate(gen=test_generator)
        ygt2, feat2 = evaluate(gen=train_generator)
        ygt3, feat3 = evaluate(gen=eval_generator)
        ygt_c= np.concatenate([ygt, ygt2, ygt3])
        ygt_t= np.concatenate([ygt, ygt3])
        feat_c= np.concatenate([feat, feat2, feat3])
        feat_t= np.concatenate([feat, feat3])
#        dsc_c = np.concatenate([dsc, dsc2, dsc3])
#        dsc_t = np.concatenate([dsc, dsc3])
        reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
        
        trans= reducer.fit(feat_c)

        embedding= reducer.fit_transform(feat_c)
        
        #embedding_t= reducer.fit_transform(feat_t)
        
#        #colors_age=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_c]
#        colors_age=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_c]
#        #colors_aget=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_t]
#        colors_aget=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_t]
        
        colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_c]
#        colorst=[sns.color_palette('hls',5)[int(i-1)] for i in ygt_t]
        #sns.palplot(sns.color_palette('hls',7))
        plt.scatter(trans.embedding_[:,0],trans.embedding_[:,1],color=colors, s=.1)
        
#        reducer3d=umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=3)
#        embedding3d = reducer3d.fit_transform(feat_c)
#        embedding3dt = reducer3d.fit_transform(feat_t)
#
#        fig = plt.figure()
#        ax= fig.add_subplot(111, projection='3d')
#        ax.scatter(embedding3d[:,0],embedding3d[:,1], embedding3d[:,2], color=colors, s=0.1)
