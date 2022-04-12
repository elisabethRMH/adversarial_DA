'''
Adversarial DA for Dreem dataset

The components here are:
    - Source net (feature extractor +  classifier)
    - Target net (feature extractor + classifier)
    - Domain discriminator
    
Losses are:
    - discriminator loss (minimax or GAN, GAN loss in final version)
    - supervised class loss (for the source data, few target recordings or both)
    - pseudo-label (or cross-entropy loss: not in final version)
    - potential MMD loss: not added in final version
    
This version; source network gets data from MASS dataset, target network gets data from the dreem dataset
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #force not using GPU!
import numpy as np
import tensorflow as tf
import math


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

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy
from subgenfromfile_epochsave import SubGenFromFile

from save_functions import *


filename="dreem_subsets_25pat.mat" #Fz or Fp2, they are in channel 5 #46pat_osaprosp
files_folds=loadmat(filename)


root = '/esat/stadiusdata/public/Dreem'
source=root+'/processed_tb/dreem25-healthy-headbandpsg2'

def train_nn(files_folds, source, config,foldrange=range(12)):


    number_patients=config.number_patients
    #VERSION WITH PATIENT GROUPS
    for fold in foldrange:
        for pat_group in range(1):#int(26/number_patients)):
    
            fileidx=np.arange(pat_group* number_patients,(pat_group+1)*number_patients)

            test_files=files_folds['test_sub'][0][fold][0]
            eval_files=files_folds['eval_sub'][0][fold][0]
            train_files_target_labeled=files_folds['train_sub'][0][fold][0][fileidx]
            train_files_target = files_folds['train_sub'][0][fold][0]
            
            # config= Config()
            config.epoch_seq_len=10
            config.epoch_step=config.epoch_seq_len
                    
            # #Both C4-A1 & new channel of sleep uzl
            test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)
            batch_size=8
            retrain_generator= SubGenFromFile(source,shuffle=True, batch_size=batch_size,subjects_list=train_files_target_labeled,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)
            eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size,  subjects_list=eval_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)
           
            
            # Target dataset: new channel of dreem
            train_generator1= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size,subjects_list=train_files_target,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)
            train_generator1.batch_size=int(np.floor(len(train_generator1.datalist)/len(retrain_generator)))
    
            # Source dataset: C4-A1 of MASS dataset
            eeg_train_data= "/esat/asterie1/scratch/ehereman/data_processing_SeqSlNet/tf_data3/seqsleepnet_eeg/train_list_total.txt"
            list1= [eeg_train_data]
            train_generator= SubGenFromFileHuy(filelist_lst=list1,shuffle=True, batch_size=config.batch_size,  sequence_size=config.epoch_seq_len,normalize_per_subject=True)
            train_generator.batch_size=train_generator1.batch_size #min(int(np.floor(len(train_generator.datalist)/len(retrain_generator))),200)
                
                
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
            
            # Load pre-trained model from this path
            tf.app.flags.DEFINE_string("out_dir1", '/esat/asterie1/scratch/ehereman/results_SeqSleepNet_tb/totalmass2/seqsleepnet_sleep_nfilter32_seq10_dropout0.75_nhidden64_att64_1chan_subjnorm/total', "Point to output directory")
            
            # Save adapted model to this path
            tf.app.flags.DEFINE_string("out_dir", config.out_dir0+'/n{:d}/group{:d}'.format( fold, pat_group), "Point to output directory")
            tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
            
            tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
            
            tf.app.flags.DEFINE_integer("seq_len", 10, "Sequence length (default: 32)")
            
            tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
            
            tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
            tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 20)")
            
            
            FLAGS = tf.app.flags.FLAGS
            print("\nParameters:")
            for attr, value in sorted(FLAGS.__flags.items()): # python3
                print("{}={}".format(attr.upper(), value))
            print("")
            
            # Data Preparatopn
            # ==================================================
            
            #save paths
            # path where some output are stored
            out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
            # path where checkpoint models are stored
            checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
            if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
            if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))
            
            #restore pre-trained model paths
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
            config.training_epoch = int(20)
            config.same_network=True #same feature extractor network for source & target
            config.learning_rate=1e-4
            config.channel = 3 #channel number corresponding to F7-F8
            config.GANloss=True #GAN loss (if false, we use a gradient reversal layer)
            config.domainclassifier=True #always put to true (unless we use MMD loss instead)
            
            #Settings specified/varied outside the training function.
            # config.pseudolabels = False
            # config.weightpslab=0.1
            # config.minneighbordiff=False
            # config.withtargetlabels=False
            # config.domain_lambda= 0.1
           
            #Settings/functionalities not used in final version of paper
            config.crossentropy=False #crossentropy loss: alternative for pseudo-labels
            config.shareDC=False #shared domain classifier for the full sequence
            config.shareLC=False #shared label classifier fro the full sequence
            config.mmd_loss=False #MMD loss (alternative for adversarial training, domain adaptation with matching distributions)
            config.mmd_weight=1 #weight of mmd loss
            config.add_classifieroutput=False #add classifier output to the domain classifier input
            config.fix_sourceclassifier=False
            
            train_batches_per_epoch = np.floor(len(retrain_generator)).astype(np.uint32)
            eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
            test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
            
            #E: nb of epochs in each set (in the sense of little half second windows)
            print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator1.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
            
            #E: nb of batches to run through whole dataset = nb of sequences divided by batch size
            print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
            
            
            
            # variable to keep track of best performance
            best_fscore = 0.0
            best_acc = 0.0
            best_loss=np.inf
            best_kappa = 0.0
            min_loss = float("inf")
            best_train_total_loss=np.inf
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
            
                    # Define Training procedure
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    optimizer = tf.train.AdamOptimizer(config.learning_rate)
                    
                    
                    domainclass_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'domainclassifier_net')
                    if config.fix_sourceclassifier:
                        excl= [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'seqsleepnet_source') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'output_layer/output-%s') ]
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
                    
                    if not config.same_network: # 2 separate feature extractors
                        sess.run(tf.initialize_all_variables())
    #                    ##REstore
    #                    best_dir = os.path.join(checkpoint_path, "best_model_acc")
    #                    saver.restore(sess, best_dir)
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
                        var_list2= {}
                        for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target'):
                            tmp=v2.name[v2.name.find('/')+1:-2]
                            var_list2[tmp]=v2
                        saver2=tf.train.Saver(var_list=var_list2)
                        saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                        saver1 = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                    else:          #shared feature extractor           
                        # initialize all variables
                        sess.run(tf.initialize_all_variables())
    #                    ##REstore
    #                    best_dir = os.path.join(checkpoint_path, "best_model_acc")
    #                    saver.restore(sess, best_dir)
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
                            # if tmp[0]=='2':
                            #     continue
                            var_list2[tmp]=v2
                        saver2=tf.train.Saver(var_list=var_list2)
                        saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                        
    
                        # saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer'),max_to_keep=1)
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
                        if config.domainclassifier and config.GANloss:
                            _,_, step, output_loss, total_loss, accuracy = sess.run(
                               [train_op, train_op2, global_step, arnn.output_loss, arnn.loss, arnn.accuracy],
                               feed_dict)
                        else:
                            _, step, output_loss, total_loss, accuracy = sess.run(
                               [train_op, global_step, arnn.output_loss, arnn.loss, arnn.accuracy],
                               feed_dict)
                            
    
                        return step, output_loss, total_loss, np.mean(accuracy)
            
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
                        if config.domainclassifier:
                            output_loss, domain_loss, total_loss, yhat, yhattarget, yhatD = sess.run(
                                   [arnn.output_loss, arnn.domain_loss_sum, arnn.loss, arnn.predictions, arnn.predictions_target, arnn.predictions_D], feed_dict)
                            return output_loss, domain_loss, total_loss, yhat, yhattarget, yhatD
                        elif config.mmd_loss:
                            output_loss, domain_loss, total_loss, yhat, yhattarget, yhatD = sess.run(
                                   [arnn.output_loss, arnn.mmd_loss, arnn.loss, arnn.predictions, arnn.predictions_target, arnn.predictions_D], feed_dict)
                            return output_loss, domain_loss, total_loss, yhat, yhattarget, yhatD
            
                            
            
    
                    def evaluate(gen, log_filename):
                        # Validate the model on the entire evaluation test set after each epoch
                        datalstlen=len(gen.datalist)
                        output_loss =0
                        total_loss = 0
                        domain_loss=0
                        yhat = np.zeros([config.epoch_seq_len, datalstlen])
                        yhattarget = np.zeros([config.epoch_seq_len, datalstlen])
                        num_batch_per_epoch = len(gen)
                        test_step = 0
                        ygt = np.zeros([config.epoch_seq_len, datalstlen])
                        yhatD = np.zeros([config.epoch_seq_len, datalstlen*2])
                        yd= np.zeros([config.epoch_seq_len, datalstlen*2])
                    
                        while test_step < num_batch_per_epoch-1:
    
    
                            (x_batch,y_batch)=gen[test_step]
                            x_batch=x_batch[:,:,:,:,[0, config.channel]]
    
            
                            output_loss_, domain_loss_, total_loss_, yhat_, yhattarget_, yhatD_ = dev_step(x_batch, y_batch)
                            ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                            yd[:,(test_step)*config.batch_size*2 : (test_step+1)*config.batch_size*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                            for n in range(config.epoch_seq_len):
                                yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                                yhattarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhattarget_[n]
                                yhatD[n, (test_step)*config.batch_size*2 : (test_step+1)*config.batch_size*2] = yhatD_[n]
                            output_loss += output_loss_
                            total_loss += total_loss_
                            domain_loss += domain_loss_
                            test_step += 1

                                
                        if len(gen.datalist) > test_step*config.batch_size:
                            (x_batch,y_batch)=gen.get_rest_batch(test_step)
    
                            x_batch=x_batch[:,:,:,:,[0, config.channel]]
            
                            output_loss_, domain_loss_, total_loss_, yhat_, yhattarget_, yhatD_ = dev_step(x_batch, y_batch)
                            ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                            yd[:,(test_step)*config.batch_size*2 : len(gen.datalist)*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                            for n in range(config.epoch_seq_len):
                                yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                                yhattarget[n, (test_step)*config.batch_size : len(gen.datalist)] = yhattarget_[n]
                                yhatD[n, (test_step)*config.batch_size*2 : len(gen.datalist)*2] = yhatD_[n]
                            output_loss += output_loss_
                            domain_loss += domain_loss_
                            total_loss += total_loss_
                        yhat = yhat + 1
                        ygt= ygt+1
                        yhattarget+=1
                        acc = accuracy_score(ygt.flatten(), yhat.flatten())
                        acctarget= accuracy_score(ygt.flatten(), yhattarget.flatten())
                        accD = accuracy_score(yd.flatten(), (yhatD>0.5).flatten())
    
                        with open(os.path.join(out_dir, log_filename), "a") as text_file:
                            text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} \n".format(output_loss, domain_loss, total_loss, acc, acctarget, accD))
                        return acctarget, yhat, output_loss, total_loss
    
                    
                    start_time = time.time()
                    time_lst=[]
            
                    # Loop over number of epochs
                    eval_acc, eval_yhat, eval_output_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                    test_acc, test_yhat, test_output_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
                    for epoch in range(config.training_epoch):
                        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                        step = 0
                        while step < train_batches_per_epoch:
                            # Get a batch
                            #((x_batch, y_batch),_) = train_generator[step]
                            t1=time.time()        
                            #Batch 3 : C4-A1 of mass
                            (x_batch3,y_batch3)=train_generator[step]
                            x_batch3=np.append(x_batch3,np.zeros(x_batch3.shape),axis=-1)
                            # #batch 1: target channel of Dreem
                            (x_batch1,y_batch1)=train_generator1[step]
                            x_batch1=x_batch1[:,:,:,:,[config.channel]]                        
                            x_batch1=np.append(np.zeros(x_batch1.shape),x_batch1,axis=-1)
                            if config.withtargetlabels:
                                #batch 2: c4a1& target channel of dreem
                                (x_batch2,y_batch2)=retrain_generator[step]
                                x_batch2=x_batch2[:,:,:,:,[ 0,config.channel]]
                                # x_batch2=np.append(np.zeros(x_batch2.shape),x_batch2,axis=-1)
                                x_batch0=np.vstack([x_batch2,x_batch1, x_batch3]) #X_batch
                                y_batch0=np.vstack([y_batch2,np.zeros(y_batch1.shape), y_batch3])#np.zeros(y_batch3.shape)])  #y_batch
                            else:
                                x_batch2= np.array([])
                                y_batch2=np.array([])
                                x_batch0=np.vstack([x_batch1, x_batch3]) #X_batch
                                y_batch0=np.vstack([np.zeros(y_batch1.shape), y_batch3])  #y_batch
                            
                            
                            t2=time.time()
                            time_lst.append(t2-t1)                        
                            target_bool=np.concatenate([np.ones(len(x_batch2)), np.ones(len(x_batch1)),  np.zeros(len(x_batch3))])#np.zeros(len(x_batch)),
                            source_bool = np.concatenate([np.ones(len(x_batch2)), np.zeros(len(x_batch1)), np.ones(len(x_batch3))])#np.ones(len(x_batch)), 
                            
                            train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x_batch0, y_batch0, target_bool, source_bool)
    
                            
                            time_str = datetime.now().isoformat()
            
                            print("{}: step {}, output_loss {},  total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, train_acc_))
                            step += 1
            
                            current_step = tf.train.global_step(sess, global_step)
                            if current_step % config.evaluate_every == 0:
                                # Validate the model on the entire evaluation test set after each epoch
                                print("{} Start validation".format(datetime.now()))
                                eval_acc, eval_yhat, eval_output_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                                test_acc, test_yhat, test_output_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")
            
                                if train_total_loss_ < best_train_total_loss:
                                    checkpoint_name1 = os.path.join(checkpoint_path, 'model_step_TRAIN' + str(current_step) +'.ckpt')
                                    save_path1 = saver1.save(sess, checkpoint_name1)
                                    source_file = checkpoint_name1
                                    dest_file = os.path.join(checkpoint_path, 'best_model_TRAIN')
                                    shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                    shutil.copy(source_file + '.index', dest_file + '.index')
                                    shutil.copy(source_file + '.meta', dest_file + '.meta')
                                    best_train_total_loss = train_total_loss_
                                if(eval_acc > best_acc):
                                    best_acc = eval_acc
                                    checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                                    save_path = saver.save(sess, checkpoint_name)
            
                                    print("Best model updated")
                                    source_file = checkpoint_name
                                    dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                                    shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                    shutil.copy(source_file + '.index', dest_file + '.index')
                                    shutil.copy(source_file + '.meta', dest_file + '.meta')
                                    
            
                        train_generator.on_epoch_end()
                        retrain_generator.on_epoch_end()
                        train_generator1.on_epoch_end()
                        
                    end_time = time.time()
                    with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                        text_file.write("{:g}\n".format((end_time - start_time)))
                        text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
        
                    save_neuralnetworkinfo(checkpoint_path, 'fmandclassnetwork',arnn,  originpath=__file__, readme_text=
                            'Domain adaptation unsup with GAN loss and classification network on sleep uzl Fz (Fp2 for some files) (with normalization per patient),\n subject normalization \n source net and target net are different, initialized with SeqSleepNet trained on mass\n'+
                            'training on {:d} patients \n baseline net is trained on 190 pat \n batch size 149+8, WITH target classifier, early stop at best eval acc on target. LR 1e-4 \n'.format(number_patients)+
                            'in this version, more data (48 pat). unfixed source network\n unlabeled unmatched C4-A1 data from mass \n using only mass data in source classifier (all patients), only C4-A1 of sleepuzl in target classifier (all patients). different patients source/target. \n\n'+
                            print_instance_attributes(config))

config=Config()

# config.domain_lambda= 0.01
# config.weightpslab=0.01
# config.pseudolabels = False
# config.number_patients=2
# config.withtargetlabels=False
# config.minneighbordiff=False
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_fnl-hyperparexp20_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config)

# config.domain_lambda= 0.01
# config.weightpslab=0.01
# config.pseudolabels = True
# config.number_patients=2
# config.withtargetlabels=False
# config.minneighbordiff=False
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps001_pslab01-2_fnl-hyperparexp20_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config)#range(5,12))

config.domain_lambda= 0.01
config.weightpslab=0.01
config.pseudolabels = False
config.number_patients=2
config.withtargetlabels=True
config.minneighbordiff=False
config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_withtargetlab_fnl-hyperparexp20_subjnorm_fz{:d}pat'.format(config.number_patients)
train_nn(files_folds, source, config)

# config.domain_lambda= 0.01
# config.weightpslab=0.1*10/21
# config.pseudolabels = True
# config.number_patients=2
# config.withtargetlabels=True
# config.minneighbordiff=False
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps01adapt_pslab01-2_withtargetlab_hyperparexp2-1_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config,foldrange=range(3,6))

# config.domain_lambda= 0.01
# config.weightpslab=0.1
# config.pseudolabels = True
# config.number_patients=2
# config.withtargetlabels=False
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps01_pslab01-2_hyperparexp2-1_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config)


# config.domain_lambda= 0.01
# config.weightpslab=0.1
# config.pseudolabels = True
# config.withtargetlabels=True
# config.number_patients=2
# config.minneighbordiff=False
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps01_pslab01-2_withtargetlab_hyperparexp2-1_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config)


# config.domain_lambda= 0.01
# config.weightpslab=0.01
# config.pseudolabels = True
# config.number_patients=2
# config.withtargetlabels=False
# config.minneighbordiff=True
# config.out_dir0 = '/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_lambdaps001_minneighbordiff_pslab01-2_hyperparexp2-1_subjnorm_fz{:d}pat'.format(config.number_patients)
# train_nn(files_folds, source, config)
