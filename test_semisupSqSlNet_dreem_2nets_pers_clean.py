'''
Testing loop for results from adversarial domain adaptation with
train_semisupSqSlNet_dreem_2nets_pers_clean.py
train_semisupSqSlNet_dreem_2nets_pers_BN_clean.py

best_model_TRAIN is the result saved at the lowest pseudo-label loss
best_model_acc is the result saved at the end of training

'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #force not using GPU!
import numpy as np
import tensorflow as tf
import math
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from sklearn.decomposition import PCA
import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat
import copy
import umap
import seaborn as sns

#from arnn_sleep_sup import ARNN_Sleep
from adversarialnetwork_SeqSlNet_2nets_clean import AdversarialNet_SeqSlNet_2nets
from ada_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap
import matplotlib.pyplot as plt

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *
from distribution_comparison import distribution_differences
sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_epochsave import SubGenFromFile

filename="./dreem_subsets_25pat.mat" #Fz or Fp2, they are in channel 5 #46pat_osaprosp

files_folds=loadmat(filename)

normalize=True

root = '/esat/stadiustempdatasets/ehereman/dreemdata'
root = '/esat/stadiusdata/public/Dreem'
source=root+'/processed_tb/dreem25-healthy-headbandpsg2'

number_patients=2

ind=0
for fold in range(12):
    for pat_group in range(len(files_folds['test_sub'][0][fold][0])):#int(26/number_patients)):
                

        test_files=[files_folds['test_sub'][0][fold][0][pat_group]]
    
        config= Config()
        config.epoch_seq_len=10
        config.epoch_step=config.epoch_seq_len
        config.subjectclassifier_trg = False
        config.subjectclassifier_src = False
        
        
    
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)

        
            
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
        
#        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_subjnorm_totalmass2/n{:d}'.format( fold), "Point to output directory")
        dir2= '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/personalization/seqslnet_advDA_fzfp2tofzfp2_fromTL_samenetwork_all_fixSN_fixedbatch32_noKL_pslab_BN2_10epevalevery100_4_shuffle3_subjnorm_fz{:d}pat'.format(number_patients, fold, pat_group)
        # dir2='/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/personalization/seqslnet_advDA_fzfp2tofzfp2_fromTL_samenetwork_all_fixSN_fixedbatch32_KL_BN1-2_minneighbordiff_pslab01-2_10epevalevery100_shuffle3_subjnorm_fz2pat'.format(number_patients, fold, pat_group)
        # dir2= '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/personalization/seqslnet_advDA_fzfp2tofzfp2_fromTL_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps001_noKL_pslab01-2_10epevalevery100_4_shuffle3_subjnorm_fz{:d}pat'.format(number_patients, fold, pat_group)
        tf.app.flags.DEFINE_string("out_dir", dir2+'/n{:d}/group{:d}'.format( fold, pat_group), "Point to output directory")
        tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
        
        tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
        
        tf.app.flags.DEFINE_integer("seq_len", 10, "Sequence length (default: 32)")
        
        tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
        
        tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
        tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 20)")
        
        
        tf.app.flags.DEFINE_integer('D',100,'Number of features') #new flag!
        
        FLAGS = tf.app.flags.FLAGS
        print("\nParameters:")
        for attr, value in sorted(FLAGS.__flags.items()): # python3
            print("{}={}".format(attr.upper(), value))
        print("")
        
        # Data Preparatopn
        # ==================================================
        
        # path where some output are stored
        out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        # path where checkpoint models are stored
        checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
        if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

#        out_path1 = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir1))
#        out_path1= FLAGS.out_dir1 #os.path.join(FLAGS.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        # path where checkpoint models are stored
#        checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
#        if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
#        if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
        
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.training_epoch = int(20) #/6 if using load_random_tuple
        config.same_network=True
        config.feature_extractor=True
        config.learning_rate=1e-4
        config.mse_weight=1.0
#        config.mse_weight=config.mse_weight*(train_generator.batch_size)/(retrain_generator.batch_size)
        config.mult_channel=False
        config.withtargetlabels=False
        config.channel = 3
        config.add_classifieroutput=False
        config.GANloss=True
        config.domain_lambda= 0.01
        config.fix_sourceclassifier=False
        config.domainclassifier=True
        config.shareDC=False
        config.shareLC=False
        config.mmd_loss =False
        config.mmd_weight=1
        config.DCweighting=False
        config.SNweighting=False
        config.pseudolabels = True
        config.DCweightingpslab=False
        config.SNweightingpslab=False
        config.weightpslab=1
        config.crossentropy=False
        config.classheads2=False
        config.advdropout=False
        config.adversarialentropymin=False
        config.minneighbordiff=False
        config.nb_subjects= 38#len(train_generator1.subjects_list)
        config.subject_lambda=0.01
        config.diffattn=False
        config.diffepochrnn=False
        config.regtargetnet=False
        config.KLregularization=False
                        
        
#        train_batches_per_epoch = np.floor(len(retrain_generator)).astype(np.uint32)
        # eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
#        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
#        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        
        
        
        # variable to keep track of best fscore
        best_fscore = 0.0
        best_acc = 0.0
        best_loss=np.inf
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
                arnn=AdversarialNet_SeqSlNet_2nets(config)
                

        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

                best_dir = os.path.join(checkpoint_path, "best_model_TRAIN")
                saver.restore(sess, best_dir)
                print("Model loaded")
        
        
        
                def evalfeatures_target(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                        arnn.target_bool:np.ones(len(x_batch)),
                        arnn.source_bool: np.ones(len(x_batch)),# np.ones(len(x_batch)),
                        arnn.input_x: x_batch,
                        arnn.input_y: y_batch,
                        arnn.dropout_keep_prob_rnn: 1.0,
                        arnn.frame_seq_len: frame_seq_len,
                        arnn.epoch_seq_len: epoch_seq_len,
                        arnn.training:False,
                        arnn.weightpslab:config.weightpslab
                    
                    }
                    if config.subjectclassifier_src or config.subjectclassifier_trg:
                        feed_dict[arnn.subject]=np.zeros((len(y_batch),config.nb_subjects))               # output_loss, mse_loss, total_loss, yhat, yhattarget = sess.run(
                    output_loss, total_loss, yhat, yhattarget, score, scoretarget, features1, features2 = sess.run(
                           [arnn.output_loss, arnn.loss, arnn.predictions, arnn.predictions_target, arnn.scores, arnn.scores_target, arnn.features1, arnn.features2], feed_dict)
                    return output_loss,total_loss, yhat, yhattarget, score, scoretarget, features1, features2
   
                def evaluate(gen):
                    # Validate the model on the entire evaluation test set after each epoch
        
                    output_loss =0
                    total_loss = 0
                    yhat = np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    yhattarget =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    score = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    scoretarget = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    featC = np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    feattarget= np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    while test_step < num_batch_per_epoch-1:
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch,y_batch)=gen[test_step]
                        x_batch=x_batch[:,:,:,:,[0, config.channel]]
        
                        output_loss_, total_loss_, yhat_, yhat2, score_, scoretarget_, features1_,features2_ = evalfeatures_target(x_batch, y_batch)
                        output_loss += output_loss_
                        total_loss += total_loss_
                        
                        featC[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features1_)
                        feattarget[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features2_)

                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2[n]
                            score[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = score_[n]
                            scoretarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = scoretarget_[n]

                        test_step += 1
                            
                    if len(gen.datalist) > test_step*config.batch_size:
                        # if using load_random_tuple
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch, y_batch) = gen.get_rest_batch(test_step)
                        x_batch=x_batch[:,:,:,:,[0, config.channel]]
        
                        output_loss_, total_loss_, yhat_, yhat2, score_, scoretarget_, features1_,features2_ = evalfeatures_target(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat2[n]
                            score[n, (test_step)*config.batch_size : len(gen.datalist),:] = score_[n]
                            scoretarget[n, (test_step)*config.batch_size : len(gen.datalist),:] = scoretarget_[n]

                        featC[:,:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features1_)
                        feattarget[:,:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features2_)
        
                        output_loss += output_loss_
                        total_loss += total_loss_
                    yhat = yhat + 1
                    ygt= ygt+1
                    yhattarget+=1
                    acc = accuracy_score(ygt.flatten(), yhat.flatten())
                    print(acc)
                    acc1 = accuracy_score(ygt.flatten(), yhattarget.flatten())
                    print(acc1)
                    return featC, feattarget, ygt, yhat, yhattarget,score, scoretarget


#
#        
                print('Test')
                feat_c1, feat_target1, ygt1, yyhatc341, yyhattarget1, scorec341, scoretarget1 = evaluate(gen=test_generator)
#                 print('Train')
# #                feat_c2, feat_target2, ygt2, yyhatc342, yyhattarget2, scorec342, scoretarget2 = evaluate(gen=train_generator)
                print('Eval')
                # feat_c3, feat_target3, ygt3, yyhatc343, yyhattarget3, scorec343, scoretarget3  = evaluate(gen=eval_generator)
#                 print('retrain')
                # feat_c4, feat_target4, ygt4, yyhatc344, yyhattarget4, scorec344, scoretarget4  = evaluate(gen=retrain_generator)
                # feat_c4, feat_target4, ygt4, yyhatc344, yyhattarget4, scorec344, scoretarget4  = evaluate(gen=train_generator)
        
                savemat(os.path.join(out_path, "test_ret_TR.mat"), dict(yhat = yyhattarget1, acc = accuracy_score(ygt1.flatten(), yyhattarget1.flatten()),kap=kap(ygt1.flatten(), yyhattarget1.flatten()),
                                                                      ygt=ygt1, subjects=test_generator.subjects_datalist, score=scoretarget1, scorec34=scorec341)  )              
