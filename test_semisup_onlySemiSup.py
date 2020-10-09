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

from adversarialnetwork import AdversarialNetwork
from adversarialnet_config import Config

#from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#from sklearn.metrics import cohen_kappa_score
#
#from datagenerator_from_list_v2 import DataGenerator
#
sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_adversarialV import SubGenFromFile

filename="/users/sista/ehereman/Documents/code/selfsup_Banville/data_split_eval.mat"
#filename='data_split_eval.mat'
#filename='data_split_eval_SS1-3.mat' # MORE data to train, same test and eval set

#filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/train_test_eval.mat'
filename='/users/sista/ehereman/GitHub/SeqSleepNet/data_processing/data_split_eval.mat'
files_folds=loadmat(filename)
#source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch2'; # no overlap
#source='/users/sista/ehereman/Desktop/all_data_epoch4'; # no overlap
source='/volume1/scratch/ehereman/processedData_toolbox/all_data_epoch_f3f4'; # no overlap

#source='/users/sista/ehereman/Documents/code/fold3_eval_data'
#root = '/esat/biomeddata/ehereman/MASS_toolbox'
config= Config()
fold=0
test_files=files_folds['test_sub'][fold][0][0]
eval_files=files_folds['eval_sub'][fold][0][0]

test_generator1 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)
test_generator2 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=1, normalize=True)
test_generator2.switch_channels(channel_to_use=1)
eval_generator3 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
eval_generator4 =SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=eval_files, sequence_size=1, normalize=True)
eval_generator4.switch_channels(channel_to_use=1)

for percent_unlabeled in [0.7,0.8, 0.9]:
    for semisup in [True]:
                
        
        
        training_epoch=40
        training_length=80000
        #percent_unlabeled=0.9
        
        config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches'
        #config.out_dir = '/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_3'
        config.checkpoint_dir= './checkpoint/'
        config.allow_soft_placement=True
        config.log_device_placement=False
        config.nchannel=1 #only for 3channel baseline!
        if semisup:
            config.out_path = os.path.join(config.out_dir, 'SEMISUP{}unlabeled'.format(percent_unlabeled))
            config.domainclassifier = True
        else:
            config.out_path = os.path.join(config.out_dir, 'FULLYSUP{}unlabeled'.format(percent_unlabeled))
            config.domainclassifier = False    
            
        #config.out_path = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
        config.checkpoint_path = os.path.abspath(os.path.join(config.out_path,config.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(config.out_path)): os.makedirs(os.path.abspath(config.out_path))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
    
        
        test_batches_per_epoch = np.floor(len(test_generator1)).astype(np.uint32)
        
        
        print("Test set: {:d}".format(len(test_generator1._indices)))
        
        print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))
        
        
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=config.allow_soft_placement,
              log_device_placement=config.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                arnn=AdversarialNetwork(config, session=sess)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
                saver = tf.train.Saver(tf.all_variables())
                # Load saved model to continue training or initialize all variables
                best_dir = os.path.join(config.checkpoint_path, "best_model_acc")
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
                    output_loss, domain_loss, total_loss, yhat, score, predD = sess.run(
                           [arnn.output_loss_mean,arnn.domain_loss_mean, arnn.loss, arnn.predictionC, arnn.score_C, arnn.predictionD], feed_dict)
                    return output_loss, domain_loss, total_loss, yhat, score, predD
        
                def evaluate(gen):
                    # Validate the model on the entire evaluation test set after each epoch
                    output_loss =0
                    domain_loss=0
                    total_loss = 0
                    yhat = np.zeros(len(gen.datalist))
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt = np.zeros(len(gen.datalist))
                    score = np.zeros((len(gen.datalist),config.nclass))
                    predictionD= np.zeros(len(gen.datalist))
                    while test_step < num_batch_per_epoch:
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch,y_batch)=gen[test_step]
                        x_batch=x_batch[:,0,:,:,0:3]
                        #x_batch=x_batch[:,0]
                        y_batch=y_batch[:,0]
                
                        output_loss_, domain_loss_, total_loss_, yhat_, score_, predD_ = dev_step(x_batch, y_batch)
                        output_loss += output_loss_
                        total_loss += total_loss_
                        domain_loss += domain_loss_
                
                        yhat[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_
                        ygt[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = y_batch
                        score[(test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = score_
                        predictionD[(test_step)*config.batch_size : (test_step+1)*config.batch_size] = predD_
                        
                        test_step += 1
    
    
        
                    if len(gen.datalist) - test_step*config.batch_size==1:
                        yhat=yhat[0:-1]
                        ygt=ygt[0:-1]
                        score = score[0:-1]
                        predictionD=predictionD[0:-1]
                    elif len(gen.datalist) > test_step*config.batch_size:
    
                        # if using load_random_tuple
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch,y_batch)=gen[test_step]
                        x_batch=x_batch[:,0,:,:,0:3]
                        y_batch=y_batch[:,0]
                
                        output_loss_, domain_loss_, total_loss_, yhat_, score_, predD_ = dev_step(x_batch, y_batch)
                        ygt[(test_step)*config.batch_size : len(gen.datalist)] = y_batch
                        yhat[(test_step)*config.batch_size : len(gen.datalist)] = yhat_
                        score[(test_step)*config.batch_size : len(gen.datalist),:] = score_
                        predictionD[(test_step)*config.batch_size : len(gen.datalist)] = predD_
                        output_loss += output_loss_
                        total_loss += total_loss_
                        domain_loss += domain_loss_
                    predictionD=predictionD<0.5
                    yhat = yhat + 1
                    ygt= ygt+1
                    acc = accuracy_score(ygt, yhat)
                    return acc, yhat, score, output_loss, domain_loss, total_loss, ygt, predictionD
        
                test_acc, test_yhat, test_score, test_output_loss, test_domain_loss, test_total_loss, ygt, predD = evaluate(gen=test_generator2)
                print(test_acc, (np.sum(predD))/len(predD))
                savemat(os.path.join(config.out_path, "test_retF34.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score,
                                                                         output_loss = test_output_loss,
                                                                     total_loss = test_total_loss, ygt=ygt))                
        
                test_acc, test_yhat, test_score, test_output_loss, test_domain_loss, test_total_loss, ygt, predD = evaluate(gen=test_generator1)
                print(test_acc, (np.sum(predD))/len(predD))
                savemat(os.path.join(config.out_path, "test_retC34.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score,
                                                                         output_loss = test_output_loss,
                                                                     total_loss = test_total_loss, ygt=ygt))
                test_acc, test_yhat, test_score, test_output_loss, test_domain_loss, test_total_loss, ygt, predD = evaluate(gen=eval_generator4)
                print(test_acc, (np.sum(predD))/len(predD))
                savemat(os.path.join(config.out_path, "eval_retF34.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score,
                                                                         output_loss = test_output_loss,
                                                                     total_loss = test_total_loss, ygt=ygt))                                
                test_acc, test_yhat, test_score, test_output_loss, test_domain_loss, test_total_loss, ygt, predD = evaluate(gen=eval_generator3)
                print(test_acc, (np.sum(predD))/len(predD))
                savemat(os.path.join(config.out_path, "eval_retC34.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score,
                                                                         output_loss = test_output_loss,
                                                                     total_loss = test_total_loss, ygt=ygt))
