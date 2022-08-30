'''
Test adversarial DA for Dreem dataset

The components here are:
    - Source net (feature extractor +  classifier)
    - Target net (feature extractor + classifier)
    - Domain discriminator
    
Losses are:
    - discriminator loss (minimax or GAN, GAN loss in final version)
    - supervised class loss (for the source data, for (few) target data or both)
    - pseudo-label loss (or cross-entropy loss. pseudo-label loss used in final version)
    - potential MMD loss: not added in final version
    
This version; source network gets data from MASS dataset, target network gets data from the dreem dataset.
channel from MASS: C4-A1
channel from dreem: F7-F8
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #force not using GPU!
import numpy as np
import tensorflow as tf
 
import sys

from scipy.io import loadmat, savemat
import umap
import seaborn as sns

from adversarialnetwork_SeqSlNet_2nets_clean import AdversarialNet_SeqSlNet_2nets
from ada_config import Config

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap
import matplotlib.pyplot as plt


sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_ReadHuyData import SubGenFromFileHuy
from subgenfromfile_epochsave import SubGenFromFile

from save_functions import *

filename="./dreem_subsets_25pat.mat" 

files_folds=loadmat(filename)

normalize=True

root = '/esat/stadiusdata/public/Dreem'
source=root+'/processed_tb/dreem25-healthy-headbandpsg2'

number_patients=2


for fold in range(12):
    for pat_group in range(1):
        
        test_files=files_folds['test_sub'][0][fold][0]
        eval_files=files_folds['eval_sub'][0][fold][0]

        config= Config()
        config.epoch_seq_len=10
        config.epoch_step=config.epoch_seq_len

        eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size,  subjects_list=eval_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)
    
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
        
        dir2='/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2toc34_samenetwork_all_unfixSN_fixedbatch32_lambdaadv001_withtargetlab_fnl-hyperparexp20_subjnorm_fz{:d}pat'.format(number_patients)
        # dir2='/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-mass/seqslnet_advDA_fzfp2tof7f8_fromf7f8mass_samenetwork_all_unfixSN_fixedbatch32_hyperparexp2-1_subjnorm_fz{:d}pat'.format(number_patients)
        # dir2='/esat/asterie1/scratch/ehereman/results_adversarialDA/dreem/25pat-dodo/seqslnet_advDA_fzfp2totarget_samenetwork_all_unfixSN_fixedbatch32_hyperparexp2-1_subjnorm_fz{:d}pat'.format(number_patients)
        
        tf.app.flags.DEFINE_string("out_dir", dir2+'/n{:d}/group{:d}'.format( fold, pat_group), "Point to output directory")#'/group{:d}'
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
        
        # path where some output are stored
        out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        # path where checkpoint models are stored
        checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
        if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

        
        config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
        config.epoch_seq_len = FLAGS.seq_len
        config.epoch_step = FLAGS.seq_len
        config.nfilter = FLAGS.nfilter
        config.nhidden1 = FLAGS.nhidden1
        config.attention_size1 = FLAGS.attention_size1
        config.nchannel = 1
        config.training_epoch = int(20) #/6 if using load_random_tuple
        config.same_network=True
        config.learning_rate=1e-4
        config.mse_weight=1.0
        config.withtargetlabels=True
        config.channel = 3
        config.add_classifieroutput=False
        config.GANloss=True
        config.domain_lambda= 0.16
        config.fix_sourceclassifier=False
        config.domainclassifier=True
        config.shareDC=False
        config.shareLC=False
        config.mmd_loss =False
        config.mmd_weight=1
        config.pseudolabels = False
        config.weightpslab=1
        config.crossentropy=False
        config.adversarialentropymin=False
        config.minneighbordiff=False
        
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Eval/Test set: {:d}/{:d}".format( len(eval_generator.datalist), len(test_generator.datalist)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Eval/Test batches per epoch: {:d}/{:d}".format( eval_batches_per_epoch, test_batches_per_epoch))
        
        
        
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
                                
                # Define Training procedure
                domainclass_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'domainclassifier_net')
                if config.fix_sourceclassifier: #TODO fix this?
                    excl= [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'seqsleepnet_source') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'output_layer/output-%s') ]
                else:
                    excl=[]
                allvars= [var for var in tf.trainable_variables() if (var not in domainclass_vars and var not in excl)]
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(arnn.loss, var_list=allvars)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                if config.domainclassifier:
                    grads_and_vars2 = optimizer.compute_gradients(arnn.domainclass_loss_sum, var_list = domainclass_vars)
                    train_op2 = optimizer.apply_gradients(grads_and_vars2, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

                best_dir = os.path.join(checkpoint_path, "best_model_acc")
                saver.restore(sess, best_dir)
                print("Model loaded")
        
        
        
                def evalfeatures_target(x_batch, y_batch):
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
                        arnn.training:False,
                        arnn.weightpslab:config.weightpslab
                    
                    }
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
                    featsource = np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    feattarget= np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    while test_step < num_batch_per_epoch-1:
                        (x_batch,y_batch)=gen[test_step]
                        x_batch=x_batch[:,:,:,:,[0, config.channel]]
        
                        output_loss_, total_loss_, yhat_, yhat2, score_, scoretarget_, features1_,features2_ = evalfeatures_target(x_batch, y_batch)
                        output_loss += output_loss_
                        total_loss += total_loss_
                        
                        featsource[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features1_)
                        feattarget[:,:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(features2_)

                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat2[n]
                            score[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = score_[n]
                            scoretarget[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = scoretarget_[n]

                        test_step += 1
                            
                    if len(gen.datalist) > test_step*config.batch_size:

                        (x_batch, y_batch) = gen.get_rest_batch(test_step)
                        x_batch=x_batch[:,:,:,:,[0, config.channel]]
        
                        output_loss_, total_loss_, yhat_, yhat2, score_, scoretarget_, features1_,features2_ = evalfeatures_target(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat_[n]
                            yhattarget[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat2[n]
                            score[n, (test_step)*config.batch_size : len(gen.datalist),:] = score_[n]
                            scoretarget[n, (test_step)*config.batch_size : len(gen.datalist),:] = scoretarget_[n]

                        featsource[:,:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(features1_)
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
                    return featsource, feattarget, ygt, yhat, yhattarget,score, scoretarget

                print('Test')
                feat_source1, feat_target1, ygt1, yyhatsource1, yyhattarget1, scoresource1, scoretarget1 = evaluate(gen=test_generator)
                print('Eval')
                feat_source3, feat_target3, ygt3, yyhatsource3, yyhattarget3, scoresource3, scoretarget3  = evaluate(gen=eval_generator)
        
                savemat(os.path.join(out_path, "test_ret.mat"), dict(yhat = yyhattarget1, acc = accuracy_score(ygt1.flatten(), yyhattarget1.flatten()),kap=kap(ygt1.flatten(), yyhattarget1.flatten()),
                                                                      ygt=ygt1, subjects=test_generator.subjects_datalist, score=scoretarget1, scoresource=scoresource1)  )              
                savemat(os.path.join(out_path, "eval_ret.mat"), dict(yhat = yyhattarget3, acc = accuracy_score(ygt3.flatten(), yyhattarget3.flatten()),kap=kap(ygt3.flatten(), yyhattarget3.flatten()),
                                                                      ygt=ygt3, subjects=eval_generator.subjects_datalist, score=scoretarget3, scoresource=scoresource3)  )               
