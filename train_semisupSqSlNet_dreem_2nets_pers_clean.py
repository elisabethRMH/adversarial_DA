'''
Like feature matching semi-supervised version but without the MSE
Difference with adversarial DA is the source & target network (2 nets) & the presence of a target classifier (this could also be removed for unsupervised DA)
So the components here are:
    - Source net (feature extractor +  classifier)
    - Target net (feature extractor + classifier)
Losses are:
    - discriminator loss (minimax or GAN)
    - supervised class loss
    - potential MMD loss

This version: both source & target net get data from the 'new dataset'. Different data for each one.
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1" #force not using GPU!
import numpy as np
import tensorflow as tf
import math
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat

#from arnn_sleep_sup import ARNN_Sleep
from adversarialnetwork_SeqSlNet_2nets_clean import AdversarialNet_SeqSlNet_2nets

from ada_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy

sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_epochsave import SubGenFromFile

filename="./dreem_subsets_25pat.mat" #Fz or Fp2, they are in channel 5 #46pat_osaprosp

files_folds=loadmat(filename)


#root = '/esat/stadiustempdatasets/ehereman/dreemdata'
root = '/esat/stadiusdata/public/Dreem'
source=root+'/processed_tb/dreem25-healthy-headbandpsg2'

number_patients=2
#VERSION WITH PATIENT GROUPS
for fold in range(12):
    for pat_group in range(len(files_folds['test_sub'][0][fold][0])):#int(26/number_patients)):

        # fileidx=np.arange(pat_group* number_patients,(pat_group+1)*number_patients)

        test_files=[files_folds['test_sub'][0][fold][0][pat_group]]
        eval_files=files_folds['eval_sub'][0][fold][0]
        train_files2 = files_folds['train_sub'][0][fold][0]#[fileidx1]

        
        config= Config()
        config.epoch_seq_len=10
        config.epoch_step=config.epoch_seq_len
        config.batch_size=32
        
        # #Both c34 & new channel of sleep uzl
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)
        batch_size=8
        retrain_generator= SubGenFromFile(source,shuffle=True, batch_size=batch_size,subjects_list=test_files, sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)
        eval_generator= test_generator#SubGenFromFile(source,shuffle=False, batch_size=config.batch_size,  subjects_list=eval_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True)
       
        # Rest of patients training data
        train_generator2= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size,subjects_list=train_files2,  sequence_size=config.epoch_seq_len,normalize_per_subject=True,file_per_subject=True)
        
        # Individual patient training data (=test data)
        train_generator1= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size,subjects_list=test_files,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)
        
        
            
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
        
        tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_transferlearning/dreem/25pat/seqslnet_transferlearning_massc34tof7f8_subjnorm_sdtrain21pat_2/n{:d}/group0'.format( fold,pat_group), "Point to output directory")
        tf.app.flags.DEFINE_string("out_dir", '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/personalization/seqslnet_advDA_fzfp2tofzfp2_fromTL_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps001_noKL_pslab01-2_10epevalevery100_4_shuffle3_subjnorm_fz{:d}pat/n{:d}/group{:d}'.format(number_patients, fold, pat_group), "Point to output directory")
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
        out_path1= FLAGS.out_dir1 #os.path.join(FLAGS.out_dir1, 'FULLYSUP{}unlabeled'.format(0.0))
        # path where checkpoint models are stored
        checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
        if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
        
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.training_epoch = int(10) #/6 if using load_random_tuple
        config.same_network=True
        config.learning_rate=1e-4
        config.evaluate_every=100#int(len(test_generator))#1
        config.channel = 3
        config.domainclassifier=True
        config.GANloss=True
        config.domain_lambda= 0.01
        config.fix_sourceclassifier=False
        config.pseudolabels = True
        config.weightpslab=0.01
        config.withtargetlabels=False
        config.minneighbordiff=False


        #Settings/functionalities not used in final version of paper
        config.crossentropy=False #crossentropy loss: alternative for pseudo-labels
        config.shareDC=False #shared domain classifier for the full sequence
        config.shareLC=False #shared label classifier fro the full sequence
        config.mmd_loss=False #MMD loss (alternative for adversarial training, domain adaptation with matching distributions)
        config.mmd_weight=1 #weight of mmd loss
        config.add_classifieroutput=False #add classifier output to the domain classifier input
                
        train_batches_per_epoch = np.floor(len(train_generator1)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        # config.evaluate_every= train_batches_per_epoch #int(100*number_patients*2/40)
        # config.checkpoint_every = config.evaluate_every
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator1.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        
        
        
        # variable to keep track of best fscore
        best_fscore = 0.0
        best_acc = 0.0
        best_train_total_loss=np.inf
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
                # arnn=AdversarialNet_SeqSlNet_2nets_advdrop(config)
        
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                
                
                domainclass_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'domainclassifier_net')
                if config.fix_sourceclassifier:
                    excl= [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'seqsleepnet_source')]# + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'output_layer/output-') ]
                else:
                    excl=[]
                allvars= [var for var in tf.trainable_variables() if (var not in domainclass_vars and var not in excl)]
                grads_and_vars = optimizer.compute_gradients(arnn.loss, var_list=allvars)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                if config.GANloss:
                    global_step2 = tf.Variable(0, name="global_step2", trainable=False)
                    grads_and_vars2 = optimizer.compute_gradients(arnn.domainclass_loss_sum, var_list = domainclass_vars)
                    train_op2 = optimizer.apply_gradients(grads_and_vars2, global_step=global_step2)

        
        
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer/output-'),max_to_keep=1)
                if not config.same_network:
                    sess.run(tf.initialize_all_variables())
                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list1 = {}
                    for v1 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'output_layer/output'):
                        tmp = v1.name.replace(v1.name[0:v1.name.index('-')],'output_layer/output')
                        tmp=tmp[:-2]
                        var_list1[tmp]=v1
                    saver1=tf.train.Saver(var_list=var_list1)                    
                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source'):
                        tmp=v2.name[16:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target'):
                        tmp=v2.name[16:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    saver1 = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                else:                    
                    # initialize all variables
                    sess.run(tf.initialize_all_variables())
                    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer/output-'),max_to_keep=1)

                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list1 = {}
                    for v1 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'output_layer/output'):
                        tmp = v1.name.replace(v1.name[0:v1.name.index('-')],'output_layer/output')
                        tmp=tmp[:-2]
                        var_list1[tmp]=v1
                    saver1=tf.train.Saver(var_list=var_list1)                    
                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source'):                        
                        tmp=v2.name[v2.name.find('/')+1:-2]
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    

                    saver1 = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                print("Model loaded")
        
                def train_step(x_batch, y_batch,target_bool, source_bool): #not adapted
                    """
                    A single training step
                    """
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                     arnn.target_bool:target_bool,
                      arnn.source_bool: source_bool,
                      arnn.input_x: x_batch,
                      arnn.input_y: y_batch,
                      arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                      arnn.frame_seq_len: frame_seq_len,
                      arnn.epoch_seq_len: epoch_seq_len,
                      arnn.training: True,
                      arnn.weightpslab:config.weightpslab
                    }
                    if config.GANloss:
                        if config.pseudolabels or config.crossentropy:
                            _,_, step, output_loss, output_loss_target, domain_loss,total_loss, accuracy = sess.run(
                               [train_op, train_op2, global_step, arnn.output_loss, arnn.output_loss_target_ps, arnn.domain_loss_sum, arnn.loss, arnn.accuracy],
                               feed_dict)
                        else:
                            _,_, step, output_loss, output_loss_target, domain_loss,total_loss, accuracy = sess.run(
                               [train_op, train_op2, global_step, arnn.output_loss, arnn.output_loss_target, arnn.domain_loss_sum, arnn.loss, arnn.accuracy],
                               feed_dict)
                            
                    else:
                        if config.pseudolabels or config.crossentropy:
                            _,_, step, output_loss, output_loss_target, domain_loss,total_loss, accuracy = sess.run(
                               [train_op, train_op2, global_step, arnn.output_loss, arnn.output_loss_target_ps, arnn.domain_loss_sum, arnn.loss, arnn.accuracy],
                               feed_dict)
                        else:
                            _, step, output_loss,output_loss_target, domain_loss, total_loss, accuracy = sess.run(
                               [train_op, global_step, arnn.output_loss,arnn.output_loss_target, arnn.domain_loss_sum, arnn.loss, arnn.accuracy],
                               feed_dict)
                    return step, output_loss,output_loss_target, domain_loss, total_loss, np.mean(accuracy)
        
                def dev_step(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                        arnn.target_bool:np.ones(len(x_batch)),
                        arnn.source_bool: np.ones(len(x_batch)),                        
                        arnn.input_x: x_batch,
                        arnn.input_y: y_batch,
                        arnn.dropout_keep_prob_rnn: 1.0,
                        arnn.frame_seq_len: frame_seq_len,
                        arnn.epoch_seq_len: epoch_seq_len,
                        arnn.training: False,
                        arnn.weightpslab:config.weightpslab
                    }
                    # output_loss, mse_loss, total_loss, yhat, yhattarget = sess.run(
                    #        [arnn.output_loss, arnn.mse_loss, arnn.loss, arnn.predictions, arnn.predictions_target], feed_dict)
                    # return output_loss, mse_loss, total_loss, (yhat), yhattarget
                    if config.pseudolabels or config.crossentropy:
                        output_loss, output_loss_target, domain_loss, total_loss, yhat, yhattarget, yhatD = sess.run(
                               [arnn.output_loss, arnn.output_loss_target_ps, arnn.domain_loss_sum, arnn.loss, arnn.predictions, arnn.predictions_target, arnn.predictions_D], feed_dict)
                    else:
                        output_loss, output_loss_target, domain_loss, total_loss, yhat, yhattarget, yhatD = sess.run(
                               [arnn.output_loss, arnn.output_loss_target, arnn.domain_loss_sum, arnn.loss, arnn.predictions, arnn.predictions_target, arnn.predictions_D], feed_dict)
                    return output_loss, output_loss_target, domain_loss, total_loss, yhat, yhattarget, yhatD
        
        
#                def evaluate(gen, log_filename):
#                    # Validate the model on the entire evaluation test set after each epoch
#        
#                    output_loss =0
#                    total_loss = 0
#                    mse_loss=0
#                    yhat = np.zeros(len(gen.datalist))
#                    test_step = 0
#                    ygt = np.zeros(len(gen.datalist))
#                    (x_batch, y_batch) = (gen.X,gen.y)
#    
#                    output_loss_, mse_loss_, total_loss_, yhat_, yhattarget_ = dev_step(x_batch, y_batch)
#                    output_loss += output_loss_
#                    total_loss += total_loss_
#                    mse_loss+= mse_loss_
#                    ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
#                    for n in range(config.epoch_seq_len):
#                        yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
#                    
#                    yhat[:] = yhat_
#                    ygt[:] = np.transpose(np.argmax(y_batch,axis=2))
#                    test_step += 1
#                    yhat = yhat + 1
#                    yhattarget= yhattarget_+1
#                    ygt= ygt+1
#                    acctarget = accuracy_score(ygt, yhattarget)
#                    acc = accuracy_score(ygt, yhat)
#                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
#                        text_file.write("{:g} {:g} {:g} {:g} {:g}\n".format(output_loss,mse_loss, total_loss, acc, acctarget))
#                    return acctarget, yhat, output_loss, mse_loss, total_loss

                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
                    datalstlen=len(gen.datalist)
                    output_loss =0
                    total_loss = 0
                    domain_loss=0
                    output_loss_target=0
                    yhat = np.zeros([config.epoch_seq_len, datalstlen])
                    yhattarget = np.zeros([config.epoch_seq_len, datalstlen])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt = np.zeros([config.epoch_seq_len, datalstlen])
                    yhatD = np.zeros([config.epoch_seq_len, datalstlen*2])
                    yd= np.zeros([config.epoch_seq_len, datalstlen*2])
                
                    while test_step < num_batch_per_epoch-1:
                        #((x_batch, y_batch),_) = gen[test_step]


                        (x_batch,y_batch)=gen[test_step]
                        x_batch=x_batch[:,:,:,:,[config.channel, config.channel]]

        
                        output_loss_,output_loss_target_, domain_loss_, total_loss_, yhat_, yhattarget_, yhatD_ = dev_step(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        yd[:,(test_step)*config.batch_size*2 : (test_step+1)*config.batch_size*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhattarget_[n]
                            yhatD[n, (test_step)*config.batch_size*2 : (test_step+1)*config.batch_size*2] = yhatD_[n]
                        output_loss += output_loss_
                        total_loss += total_loss_
                        domain_loss += domain_loss_
                        output_loss_target += output_loss_target_
                        test_step += 1
#                    if len(gen.datalist) - test_step*config.batch_size==1:
#                        yhat=yhat[0:-1]
#                        ygt=ygt[0:-1]
                            
                    if len(gen.datalist) > test_step*config.batch_size:
                        # if using load_random_tuple
                        #((x_batch, y_batch),_) = gen[test_step]
                        (x_batch,y_batch)=gen.get_rest_batch(test_step)

                        x_batch=x_batch[:,:,:,:,[config.channel, config.channel]]
        
                        output_loss_, output_loss_target_, domain_loss_, total_loss_, yhat_, yhattarget_, yhatD_ = dev_step(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        yd[:,(test_step)*config.batch_size*2 : len(gen.datalist)*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : len(gen.datalist)] = yhattarget_[n]
                            yhatD[n, (test_step)*config.batch_size*2 : len(gen.datalist)*2] = yhatD_[n]
                        output_loss += output_loss_
                        domain_loss += domain_loss_
                        total_loss += total_loss_
                        output_loss_target += output_loss_target_
                    yhat = yhat + 1
                    ygt= ygt+1
                    yhattarget+=1
                    acc = accuracy_score(ygt.flatten(), yhat.flatten())
                    acctarget= accuracy_score(ygt.flatten(), yhattarget.flatten())
                    accD = accuracy_score(yd.flatten(), (yhatD>0.5).flatten())

                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} {:g} \n".format(output_loss, output_loss_target, domain_loss, total_loss, acc, acctarget, accD))
                    return acctarget, yhat, output_loss, output_loss_target, total_loss

                
                start_time = time.time()
                time_lst=[]
        
                # Loop over number of epochs
                # eval_acc, eval_yhat, eval_output_loss,eval_output_loss_target, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                # test_acc, test_yhat, test_output_loss,test_output_loss_target, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
                for epoch in range(config.training_epoch):
                    if epoch>=10 and config.weightpslab==0:
                        config.weightpslab==0.1
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:

                        current_step = tf.train.global_step(sess, global_step)
                        
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            eval_acc, eval_yhat, eval_output_loss,eval_output_loss_target, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                            #test_acc, test_yhat, test_output_loss,eval_output_loss_target, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
                            
                            if eval_output_loss_target < best_train_total_loss: #train_total_loss_
                                checkpoint_name1 = os.path.join(checkpoint_path, 'model_step_TRAIN' + str(current_step) +'.ckpt')
                                save_path1 = saver1.save(sess, checkpoint_name1)
                                source_file = checkpoint_name1
                                dest_file = os.path.join(checkpoint_path, 'best_model_TRAIN')
                                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                shutil.copy(source_file + '.index', dest_file + '.index')
                                shutil.copy(source_file + '.meta', dest_file + '.meta')
                                best_train_total_loss = eval_output_loss_target

                        # Get a batch
                        #((x_batch, y_batch),_) = train_generator[step]
                        t1=time.time()        
                        #Batch - : c34 of mass
                        # (x_batch,y_batch)=train_generator[step]
                        # x_batch=np.append(x_batch,np.zeros(x_batch.shape),axis=-1)
                        #batch 3: c34 of sleep uzl
                        (x_batch3,y_batch3)=train_generator2[step]
                        x_batch3=x_batch3[:,:,:,:,[config.channel]]
                        x_batch3=np.append(x_batch3,np.zeros(x_batch3.shape),axis=-1)
                        # #batch 1: fz of sleep uzl
                        (x_batch1,y_batch1)=train_generator1[step]
                        x_batch1=x_batch1[:,:,:,:,[config.channel]]                        
                        x_batch1=np.append(np.zeros(x_batch1.shape),x_batch1,axis=-1)
                        if config.withtargetlabels:
                            #batch 2: c34&fz of sleepuzl
                            (x_batch2,y_batch2)=retrain_generator[step]
                            x_batch2=x_batch2[:,:,:,:,[0, config.channel]]
                            x_batch0=np.vstack([x_batch2,x_batch1, x_batch3]) #X_batch
                            y_batch0=np.vstack([y_batch2,np.zeros(y_batch1.shape), y_batch3])  #y_batch
                        else:
                            x_batch2= np.array([])
                            y_batch2=np.array([])
                            x_batch0=np.vstack([x_batch1, x_batch3]) #X_batch
                            y_batch0=np.vstack([np.zeros(y_batch1.shape), y_batch3])  #y_batch
                        
                        #Stack all the batches in one batch. 
                        # x_batch0=np.vstack([x_batch2,x_batch,x_batch1])
                        # y_batch0=np.vstack([y_batch2,y_batch,y_batch1])
                        
                        t2=time.time()
                        time_lst.append(t2-t1)                        
                        target_bool=np.concatenate([np.ones(len(x_batch2)), np.ones(len(x_batch1)),  np.zeros(len(x_batch3))])#np.zeros(len(x_batch)),
                        source_bool = np.concatenate([np.ones(len(x_batch2)), np.zeros(len(x_batch1)), np.ones(len(x_batch3))])#np.ones(len(x_batch)),  #normally x_batch1 here zeros, but for sake of KLdiv. I do 1
                        
                        train_step_, train_output_loss_, train_output_loss_target_,train_domain_loss_, train_total_loss_, train_acc_ = train_step(x_batch0, y_batch0, target_bool, source_bool)

                        
                        time_str = datetime.now().isoformat()
        
                        print("{}: step {}, output_loss {}, output_loss_target {}, domain_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_output_loss_target_, train_domain_loss_, train_total_loss_, train_acc_))
                        step += 1
        
                                
                            # if(eval_acc > best_acc):
                            #     best_acc = eval_acc
                            #     checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                            #     save_path = saver.save(sess, checkpoint_name)
        
                            #     # checkpoint_name1 = os.path.join(checkpoint_path, 'model_step_outputlayer' + str(current_step) +'.ckpt')
                            #     # save_path1 = saver1.save(sess, checkpoint_name1)
        
                            #     print("Best model updated")
                            #     source_file = checkpoint_name
                            #     dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                            #     shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            #     shutil.copy(source_file + '.index', dest_file + '.index')
                            #     shutil.copy(source_file + '.meta', dest_file + '.meta')
                                
                                # source_file = checkpoint_name1
                                # dest_file = os.path.join(checkpoint_path, 'best_model_outputlayer_acc')
                                # shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                # shutil.copy(source_file + '.index', dest_file + '.index')
                                # shutil.copy(source_file + '.meta', dest_file + '.meta')
        
                    # train_generator.on_epoch_end()
                    retrain_generator.on_epoch_end()
                    train_generator1.on_epoch_end()
                    train_generator2.on_epoch_end()
                    
                best_acc = eval_acc
                checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                # checkpoint_name1 = os.path.join(checkpoint_path, 'model_step_outputlayer' + str(current_step) +'.ckpt')
                # save_path1 = saver1.save(sess, checkpoint_name1)

                print("Best model updated")
                source_file = checkpoint_name
                dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                shutil.copy(source_file + '.index', dest_file + '.index')
                shutil.copy(source_file + '.meta', dest_file + '.meta')


                end_time = time.time()
                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                    text_file.write("{:g}\n".format((end_time - start_time)))
                    text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
    
                save_neuralnetworkinfo(checkpoint_path, 'fmandclassnetwork',arnn,  originpath=__file__, readme_text=
                        'Domain adaptation unsup personalization with GAN loss and classification network on sleep uzl Fz (Fp2 for some files) (with normalization per patient) \n source net and target net are same, initialized with SeqSleepNet trained on mass\n'+
                        'training on {:d} patients \n validation with pseudo-label accuracy \n no batch norm \n baseline net is trained on 190 pat \n batch size 32, WITH target classifier, early stop at best test pseudolabel acc on target. LR 1e-4,  \n'.format(number_patients)+
                        ' with target classification layer. not fixed source classifier. fixed batch size 32 for source and target!!!  \n'+# \n with pseudolabels: for unmatched target, we use labels from source net classifier for training target net classifier. NOT weighted. \n'+
                        '\n unlabeled unmatched fzfp2 data from sleep uzl \n using fzfp2 of sleepuzl train patients in source, only fp2fz of sleepuzl 1 pat in target classifier \n no overlap between patients source/target. \n\n'+
                        'in this version, eval every step & only 5 epochs, KL regularization wth alpha=.6. \n \n '+
                        print_instance_attributes(config))
