#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:34:04 2020

@author: ehereman
"""

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

#from arnn_sleep_sup import ARNN_Sleep
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/E2E-ARNN') #22/05: changed SeqSleepNet_E into SeqSleepNet
from arnn_sleep import ARNN_Sleep
from arnn_sleep_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
#from subgenfromfile_loadrandomtuples import SubGenFromFile
from subgenfromfile_epochsave import SubGenFromFile

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
#filename='data_split_eval.mat'
filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
files_folds=loadmat(filename)
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap
#source='/users/sista/ehereman/Desktop/all_data_epoch4'

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
tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")

tf.app.flags.DEFINE_integer("seq_len", 32, "Sequence length (default: 32)")

tf.app.flags.DEFINE_integer("nfilter", 20, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 32, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer('D',100,'Number of features') #new flag!

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

config = Config()
config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.attention_size1 = FLAGS.attention_size1
config.D = FLAGS.D #Number of features!
config.nchannel = 3
config.training_epoch = int(40) #/6 if using load_random_tuple

train_generator= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size, subjects_list=train_files, sequence_size=1, normalize=True) #TODO adapt back
eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)

X=train_generator.X
feat=X.reshape((X.shape[0],X.shape[1]*X.shape[2],X.shape[3]))
y= train_generator.y
ygt=np.argmax(y, axis=1)+1

X=eval_generator.X
feat2=X.reshape((X.shape[0],X.shape[1]*X.shape[2],X.shape[3]))
y= eval_generator.y
ygt2=np.argmax(y, axis=1)+1

X=test_generator.X
feat3=X.reshape((X.shape[0],X.shape[1]*X.shape[2],X.shape[3]))
y= test_generator.y
ygt3=np.argmax(y, axis=1)+1


ygt= np.concatenate([ygt, ygt2, ygt3])
feat= np.concatenate([feat, feat2, feat3])

feat_c=feat[:,:,0]
feat_eog= feat[:,:,1]
feat_f= feat[:,:,3]

#        dsc_c = np.concatenate([dsc, dsc2, dsc3])
#        dsc_t = np.concatenate([dsc, dsc3])
reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)

trans= reducer.fit(feat_c)
embeddingc=trans.transform(feat_c)
embeddingf=trans.transform(feat_f)
embeddingeog=trans.transform(feat_eog)

embedding_f= reducer.fit_transform(feat_f)
embedding_eog= reducer.fit_transform(feat_eog)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embeddingf[:,0], embeddingf[:,1],color=colors, s=.1)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embeddingeog[:,0], embeddingeog[:,1],color=colors, s=.1)


        #colors_age=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_c]
#colors_age=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_c]
        #colors_aget=[sns.color_palette('hls',76-18)[int(i-18)] for i in ygt_t]
#colors_aget=[sns.color_palette('hls',7)[int((i-10)/10)] for i in dsc_t]

colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt]
sns.palplot(sns.color_palette('hls',7))
plt.scatter(embeddingc[:,0],embeddingc[:,1],color=colors, s=.1)

reducer3d=umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=3)
embedding3d = reducer3d.fit_transform(feat_c)
embedding3dt = reducer3d.fit_transform(feat_t)

fig = plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(embedding3d[:,0],embedding3d[:,1], embedding3d[:,2], color=colors, s=0.1)
