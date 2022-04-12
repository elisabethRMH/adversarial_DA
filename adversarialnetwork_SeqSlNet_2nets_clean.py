#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain adaptation network using seqsleepnet for sleep staging
source & target net are different, get trained separately starting from a network trained on source domain
it has two option: adversarial domain classifier or MMD loss

version with pseudolabels (this means labels by source classifier for target classifier):

Created on Tue May 26 16:40:01 2020

@author: ehereman
"""
import sys
import tensorflow as tf
sys.path.insert(1,'/users/sista/ehereman/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet')
from nn_basic_layers import fc
# sys.path.insert(1,'/users/sista/ehereman/Documents/code/feature_mapping/')
from seqsleepnet_sleep_featureextractor import seqsleepnet_featureextractor #V2 is without the fc layer!
sys.path.insert(1,'/users/sista/ehereman/GitHub/gradient_reversal_keras_tf')
from flipGradientTF import GradientReversal #24/08/20 different implementation of flip layer. check if same
import tensorflow.keras.backend as K
# import tensorflow.keras.losses as tf.

class AdversarialNet_SeqSlNet_2nets(object):
    '''Feature mapping and classification model using seqsleepnet_featureextractor as a feature extractor network, and adds a classification layer to that
    The loss function consists of the feature map loss and classification loss
    '''
    
    def __init__(self, config):
        self.config=config
        self.source_bool=tf.placeholder(tf.float32,[None])
        self.target_bool=tf.placeholder(tf.float32,[None])
        self.input_x = tf.compat.v1.placeholder(tf.float32, [None, self.config.epoch_step, self.config.frame_step, self.config.ndim, 2], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.config.epoch_step, self.config.nclass], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32,shape=(), name="dropout_keep_prob_rnn")
        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.training=tf.placeholder(tf.bool,shape=(), name='training')
        self.weightpslab=tf.placeholder(tf.float32, shape=(),name='weightpslab')
        
        with tf.device('/gpu:0'), tf.variable_scope("seqsleepnet_source"):
            op = self.input_x[:,:,:,:,0:1]
            tmp= tf.boolean_mask(op,tf.dtypes.cast(self.source_bool, tf.bool))
            frame_tmp= tf.repeat(tf.boolean_mask(self.frame_seq_len,tf.dtypes.cast(self.source_bool, tf.bool)),self.config.epoch_seq_len, axis=0)
            epoch_tmp= tf.boolean_mask(self.epoch_seq_len,tf.dtypes.cast(self.source_bool, tf.bool))
 
            self.features1 =seqsleepnet_featureextractor(self.config,tmp, self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=False, istraining=self.training) #number=1

            #Now target feature extractor:
            tmp= tf.boolean_mask(self.input_x,tf.dtypes.cast(self.target_bool, tf.bool))
            frame_tmp= tf.repeat(tf.boolean_mask(self.frame_seq_len,tf.dtypes.cast(self.target_bool, tf.bool)),self.config.epoch_seq_len, axis=0)
            epoch_tmp= tf.boolean_mask(self.epoch_seq_len,tf.dtypes.cast(self.target_bool, tf.bool))       
            if config.same_network:
                self.features2 =seqsleepnet_featureextractor(self.config, tmp[:,:,:,:,1:2], self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=True, istraining = self.training) #number=1
                
        if not config.same_network:
            with tf.device('/gpu:0'), tf.variable_scope("seqsleepnet_target"):
                self.features2 =seqsleepnet_featureextractor(self.config, tmp[:,:,:,:,1:2], self.dropout_keep_prob_rnn, frame_tmp, epoch_tmp, reuse=False, istraining = self.training) #number=1
     
                        
            
        self.scores = []
        self.labels=[]
        self.predictions = []
        self.scores_target = []
        self.predictions_target = []  
        self.scores_targetpseudo = []
        self.predictions_targetpseudo=[]
        self.labels_target=[]
        self.labels_targetpseudo=[]
        self.onehot_targetpseudo=[]
        
        with tf.device('/gpu:0'), tf.compat.v1.variable_scope("output_layer"):
            for i in range(self.config.epoch_step):
                if i==0 or not self.config.shareLC:  
                    j=i
                    reuse=False
                else:
                    j=0
                    reuse=True
                score_i = fc((self.features1[:,i,:]),
                                self.config.nhidden2 * 2,
                                self.config.nclass,
                                name="output-%s" % j,
                                relu=False, reuse=reuse) #output: logits without softmax!
                    
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                lab_i=tf.nn.softmax(score_i,1,name='lab-%s'%i)
                self.scores.append(score_i)
                self.labels.append(lab_i)
                self.predictions.append(pred_i)

                if self.config.pseudolabels:
                    score_i_targetpseudo = fc((self.features2[:,i,:]),
                                    self.config.nhidden2 * 2,
                                    self.config.nclass,
                                    name="output-%s" % j,
                                    relu=False,reuse=True) #output: logits without softmax!
                    
                    score_i_targetpseudo=tf.stop_gradient(score_i_targetpseudo)
                    
                    pred_i_targetpseudo = tf.argmax(score_i_targetpseudo, 1, name="predtargetps-%s" % i)
                    lab_i_targetpseudo=tf.nn.softmax(score_i_targetpseudo,1,name='labtargetps-%s'%i)
                    onehot_i_targetpseudo=tf.one_hot(pred_i_targetpseudo, depth=5, name='onehottargetps-%s'%i)
                    self.scores_targetpseudo.append(score_i_targetpseudo)
                    self.labels_targetpseudo.append(lab_i_targetpseudo)
                    self.onehot_targetpseudo.append(onehot_i_targetpseudo)
                    self.predictions_targetpseudo.append(pred_i_targetpseudo)
                    
                if config.withtargetlabels or config.crossentropy or config.pseudolabels:
                    score_i_target = fc((self.features2[:,i,:]),
                                    self.config.nhidden2 * 2,
                                    self.config.nclass,
                                    name="outputtarget-%s" % j,
                                    relu=False, reuse=reuse) #output: logits without softmax!
                else:
                    score_i_target = fc((self.features2[:,i,:]),
                                    self.config.nhidden2 * 2,
                                    self.config.nclass,
                                    name="output-%s" % j,
                                    relu=False, reuse=True) #output: logits without softmax!                        
                pred_i_target = tf.argmax(score_i_target, 1, name="predtarget-%s" % i)
                lab_i_target=tf.nn.softmax(score_i_target,1,name='labtarget-%s'%i)
                self.scores_target.append(score_i_target)
                self.predictions_target.append(pred_i_target)
                self.labels_target.append(lab_i_target)
                    
        
        self.scores_D=[]
        self.predictions_D=[]
        self.accuracyD= []
        with tf.device('/gpu:0'), tf.variable_scope('domainclassifier_net'):
            for i in range(self.config.epoch_step):

                # #For vl3-1: try NOT giving the source data from MASS dataset to the domain discriminator
                # self.tmpy= tf.boolean_mask(self.input_y[:,i,:],tf.dtypes.cast(self.source_bool, tf.bool))
                # self.applicable = tf.equal(tf.reduce_sum(tf.dtypes.cast(self.tmpy,tf.int32),axis=[1]),1-tf.cast(self.training,tf.int32))
                # features1= tf.boolean_mask(self.features1[:,i,:], tf.dtypes.cast(self.applicable, tf.bool))
                # self.attention_out1= tf.concat([features1, self.features2[:,i,:]],axis=0)
                # self.domain_gt= tf.concat([tf.cast(tf.boolean_mask(self.applicable,self.applicable), tf.float32), 0*tf.boolean_mask(self.target_bool,tf.dtypes.cast(self.target_bool, tf.bool))], axis=0)
                
                self.attention_out1= tf.concat([self.features1[:,i,:], self.features2[:,i,:]],axis=0)
                self.domain_gt= tf.concat([tf.boolean_mask(self.source_bool,tf.dtypes.cast(self.source_bool, tf.bool)), 0*tf.boolean_mask(self.target_bool,tf.dtypes.cast(self.target_bool, tf.bool))], axis=0)
                if not self.config.GANloss:
                    fliplayer= GradientReversal(1.0)
                    self.domain_x1 = fliplayer(self.attention_out1)
                else:
                    self.domain_x1 = self.attention_out1
                if config.add_classifieroutput:
                    # labels_source= tf.boolean_mask(self.input_y[:,i], tf.dtypes.cast(self.source_bool, tf.bool))
                    labels_source=self.labels[i]
                    # labels_target = tf.one_hot(self.predictions_target[i], depth=5)
                    labels_target=self.labels_target[i]
                    self.domain_x0 = tf.stop_gradient(tf.concat([labels_source, labels_target],0))
                    self.domain_x1= tf.concat([self.domain_x1, self.domain_x0],1)
                    nbinputs=self.config.nhidden1 * 2 +self.config.nclass
                else:
                    nbinputs=self.config.nhidden1 * 2
                

                if i==0 or not self.config.shareDC:                    
                    j=i
                    reuse=False
                else:
                    j=0
                    reuse=True
                self.domain_x2 = fc(self.domain_x1,
                                nbinputs,
                                self.config.nfc_domainclass,
                                name="domain1-%s" % j,
                                relu=True,reuse=reuse)
                
                score_D =fc(self.domain_x2,
                                self.config.nfc_domainclass,
                                1,
                                name="outputD-%s" % j,
                                relu=False,reuse=reuse)[:,0]
                    
                predictionD = tf.math.sigmoid(score_D, name="prediction")
                result=predictionD>0.5
                self.scores_D.append(score_D)
                self.predictions_D.append(predictionD)
                # applicable = tf.not_equal(self.input_y, -1) #Only during training is accuracyD the righ tmeasure, during testing, the target samples don't have -1 as label anymore!!
                self.accuracyD.append(tf.reduce_mean(tf.to_float( tf.equal(result, tf.dtypes.cast(self.domain_gt,tf.bool)))))
        

        # calculate cross-entropy output loss
        self.output_loss = 0
        self.output_loss2 = 0
        self.output_loss_target =0# tf.constant(0)
        self.output_loss_target_ps = 0
        self.targetclasslayer_loss=0
        self.output_diff_target=0
        self.domain_loss_sum = 0
        self.domainclass_loss_sum=0
        self.kl_loss=0
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            for i in range(self.config.epoch_step):
                
                tmpy= tf.boolean_mask(self.input_y,tf.dtypes.cast(self.source_bool, tf.bool))
                mask1 = tf.boolean_mask(self.source_bool*tf.abs(tf.dtypes.cast(self.training,tf.float32)-self.target_bool), tf.dtypes.cast(self.source_bool, tf.bool))
                mask2 = tf.boolean_mask(self.source_bool*self.target_bool, tf.dtypes.cast(self.source_bool, tf.bool))
                mask3 = tf.boolean_mask(self.source_bool*self.target_bool, tf.dtypes.cast(self.target_bool, tf.bool))
                
                applicable = tf.not_equal(tf.reduce_sum(tf.dtypes.cast(tmpy,tf.int32),axis=[1,2]), 0)
                output_loss_i = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(tmpy[:,i,:]), logits=self.scores[i])            
                output_loss_i = tf.where(applicable, output_loss_i, tf.zeros_like(output_loss_i))
                tmp= tf.boolean_mask(output_loss_i,tf.dtypes.cast(mask1, tf.bool))                
                tmp2= tf.boolean_mask(output_loss_i,tf.dtypes.cast(mask2, tf.bool)) #matched source output
                output_loss_i1 = tf.reduce_sum(tmp, axis=[0])
                self.output_loss += output_loss_i1
                
                output_loss_i2= tf.reduce_sum(tmp2, axis=[0])
                self.output_loss2+= output_loss_i2
                
                if self.config.withtargetlabels:    
                    tmpy= tf.boolean_mask(self.input_y,tf.dtypes.cast(self.target_bool, tf.bool))
                    mask4=tf.dtypes.cast(tf.math.reduce_sum(tmpy[:,0],axis=(1)),tf.bool)
                    
                    self.output_loss_i_target=  tf.nn.softmax_cross_entropy_with_logits(labels=tmpy[:,i,:], logits=self.scores_target[i])
                    self.output_loss_i_target = tf.boolean_mask(self.output_loss_i_target, tf.dtypes.cast(mask4,tf.bool)) #mask3
                        
                        
                    self.output_loss_i_target = tf.reduce_sum(self.output_loss_i_target)
                    self.output_loss_target += self.output_loss_i_target
                else:
                    self.output_loss_target = tf.constant(0)
                    
                if self.config.pseudolabels or self.config.crossentropy or self.config.minneighbordiff:
                    tmpy= tf.boolean_mask(self.input_y,tf.dtypes.cast(self.target_bool, tf.bool))
                    mask4=tf.dtypes.cast(tf.math.reduce_sum(tmpy[:,0],axis=(1)),tf.bool)
                    
                    if self.config.pseudolabels:                            
                        tmpy2= self.labels_targetpseudo[i]
                        # tmpy2= self.onehot_targetpseudo[i]
                    elif self.config.crossentropy or self.config.minneighbordiff:
                        tmpy2=self.labels_target[i]
                    if self.config.minneighbordiff:
                        if i==0:
                            tmpy2prev=tmpy2
                        else:
                            _,sloss=mse_loss(tmpy2,tmpy2prev)
                            self.output_diff_target+= sloss
                            tmpy2prev=tmpy2
                    self.output_loss_i_target=  tf.nn.softmax_cross_entropy_with_logits(labels=tmpy2, logits=self.scores_target[i])
                    self.output_loss_i_target = tf.boolean_mask(self.output_loss_i_target, tf.dtypes.cast(tf.dtypes.cast(self.training,tf.float32)-tf.dtypes.cast(mask4,tf.float32),tf.bool))
                        
                    self.output_loss_i_target = tf.reduce_sum(self.output_loss_i_target)
                    self.output_loss_target_ps += self.weightpslab*self.output_loss_i_target

                    
                #Domain loss & domain classifier loss    
                self.domain_loss_sum_i, self.domain_loss_i= domainclassification_costs(labels= tf.dtypes.cast(self.domain_gt,tf.bool), logits= self.scores_D[i])
                if self.config.GANloss: #The loss for the domain classifier.
                    self.domainclass_loss_sum_i, self.domainclass_loss_i= self.domain_loss_sum_i, self.domain_loss_i
                    self.domainclass_loss_sum+= self.domain_loss_sum_i
                    if config.domainclassifier:
                        labels=tf.dtypes.cast(1-self.domain_gt,tf.bool)
                        self.domain_loss_sum_i, self.domain_loss_i= domainclassification_costs(labels= labels, logits= self.scores_D[i])
                        self.domain_loss_sum += tf.reduce_sum(self.domain_loss_i)
                    else:
                        self.domain_loss_sum=tf.constant(0)
                        self.domain_loss=tf.zeros_like(self.output_loss)
                elif config.domainclassifier:
                        self.domain_loss_sum += self.domain_loss_sum_i
                else:
                    self.domain_loss_sum=tf.constant(0)
                    self.domain_loss=tf.zeros_like(self.output_loss)
                #MMD loss
                if self.config.mmd_loss:
                    self.mmd_loss= mmd_loss(self.features1, self.features2)
                
            self.output_loss_target = self.output_loss_target/ self.config.epoch_step
            self.output_loss_target_ps = self.output_loss_target_ps/self.config.epoch_step
            self.output_loss = self.output_loss/self.config.epoch_step # average over sequence length
            self.output_loss2 = self.output_loss2/self.config.epoch_step # average over sequence length
            self.domain_loss_sum = self.domain_loss_sum /self.config.epoch_step
            self.domainclass_loss_sum = self.domainclass_loss_sum/ self.config.epoch_step
            self.output_diff_target = self.output_diff_target/self.config.epoch_step
            self.kl_loss=self.kl_loss/self.config.epoch_step
            # tmp1= tf.boolean_mask(self.features1,tf.dtypes.cast(mask2, tf.bool))
            # tmp2= tf.boolean_mask(self.features2,tf.dtypes.cast(mask3, tf.bool))    

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-eeg')
            except_vars_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-target')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_source/filterbank-layer-emg')
            except_vars_eeg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-eeg')
            except_vars_target2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-target')
            except_vars_emg2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_target/filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_target and v not in except_vars_emg
                    and v not in except_vars_eeg2 and v not in except_vars_target2 and v not in except_vars_emg2])
            
            self.loss = self.config.l2_reg_lambda*l2_loss
            
            if not self.config.fix_sourceclassifier:
                self.loss+= self.output_loss
            # if config.pseudolabels:
            #     coll2=[]
            #     coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'output_layer/outputtarget')
            #     for v in coll:
            #         tmp = v.name.replace('outputtarget','output')
            #         v2=[v1 for v1 in tf.compat.v1.global_variables() if tmp in v1.name][0]
            #         coll2.append(v-v2)
            #     self.l2_loss_outputtarget= tf.add_n([tf.nn.l2_loss(v) for v in coll2])
            #     self.loss+=self.l2_loss_outputtarget
            
            
            if self.config.withtargetlabels :
                self.loss+=1.0*(self.output_loss_target) #
            
            if self.config.pseudolabels or self.config.crossentropy:
                self.loss+= self.output_loss_target_ps
            if self.config.domainclassifier:
                self.loss+= self.domain_loss_sum *self.config.domain_lambda
            if self.config.mmd_loss:
                self.loss+= self.mmd_loss* self.config.mmd_weight
            if self.config.minneighbordiff:
                self.loss+= self.weightpslab*self.output_diff_target
            
            
                
        self.accuracy = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            for i in range(self.config.epoch_step):
                tmpy= tf.boolean_mask(self.input_y,tf.dtypes.cast(self.source_bool, tf.bool))

                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(tmpy[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)


            

def mse_loss(inputs, outputs,netactive=None,eps=None):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE, name='mse')
    
    loss= mse(outputs, inputs)
    if netactive is not None:
        loss = tf.math.multiply(netactive, loss)
#    total_count1 = tf.to_float(tf.shape(loss)[0])
#    total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(netactive, tf.int32))) #V2 adaptation Elisabeth 11/08/'20
    if eps is not None:
        loss=tf.math.max(loss,eps)
    return loss, tf.reduce_sum(loss)#*total_count1/total_count2


def flip_gradient(x, l=1.0):
    'copied from https://github.com/tachitachi/GradientReversal '
    positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
    negative_path = -x * tf.cast(l, tf.float32)
    return positive_path + negative_path   

def domainclassification_costs(logits, labels, name=None, leastsquares=False):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "domainclass_costs") as scope:
        total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(labels, tf.int32))) #V2 adaptation Elisabeth 11/08/'20
        # labels = tf.cast(labels, tf.float32)
        # This will now have incorrect values for unlabeled examples
        if leastsquares:
            mse = tf.keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE)
            per_sample= mse(tf.expand_dims(labels,1), tf.expand_dims(tf.math.sigmoid(logits),1))
        else:
            per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32))
        
        total_count1 = tf.to_float(tf.shape(per_sample)[0])

        # Take mean over all examples, not just labeled examples.
        # per_sample = tf.where(labels, per_sample*(total_count1)/total_count2/2, per_sample*(total_count1)/(total_count1-total_count2)/2)
        labeled_sum = tf.reduce_sum(per_sample)
        #total_count = tf.to_float(tf.shape(per_sample)[0])
        #mean = tf.div(labeled_sum, total_count, name=scope)

        return labeled_sum, per_sample#mean, per_sample 
       
def mmd_loss(inputs, outputs,beta=0.2):
    return tf.reduce_mean(gaussian_kernel(inputs,inputs,beta))+ tf.reduce_mean(gaussian_kernel(outputs,outputs,beta))- 2*tf.reduce_mean(gaussian_kernel(inputs,outputs,beta))
                        

def gaussian_kernel(x1, x2, beta = 0.2):
    r = tf.expand_dims(x1, 1)
    return K.exp( -beta * tf.math.reduce_sum(K.square(r - x2), axis=-1))  

# def calc_weights(domain_gt, predictions, mask=None, normalize_weights=True):
#     '''First version of DC weights: simply use 'how large is target predicted value of domain classifier' = difference with 0, its actual label
    
#     '''
#     predictions_target= tf.boolean_mask(predictions, tf.dtypes.cast(1-domain_gt, tf.bool)) #we only want to keep target which is the second part of the predictions with domain_gt=0
#     if mask is not None:
#         predictions_targetmatched= tf.boolean_mask(predictions_target, tf.dtypes.cast(mask,tf.bool)) #mask that selects the paired / matched samples with labels    
#         if normalize_weights:
#             return tf.reduce_sum(mask)*(predictions_targetmatched)/tf.reduce_sum(predictions_targetmatched) #if the GAN classifier is more confused, it's going to predict higher values for target (whereas target label is 0)
#     else: #no mask for the case where we want to select all the target samples
#         predictions_targetmatched= tf.boolean_mask(predictions_target, tf.dtypes.cast(mask,tf.bool)) #no mask
#         if normalize_weights:
#             return tf.shape(predictions_targetmatched)[0]*predictions_targetmatched/tf.reduce_sum(predictions_targetmatched)
#     return predictions_targetmatched #if no normalization, we don't scale the weights to have a sum equal to the number of samples

def calc_weights(domain_gt, predictions, mask=None, normalize_weights=True):
    '''New version of DC weights: binary cross-entropy loss of DC classifier = confusion
    '''
    bce= tf.keras.losses.BinaryCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
    crossentr=bce(tf.expand_dims(domain_gt,1), tf.expand_dims(predictions,1))
    predictions_target= tf.boolean_mask(crossentr, tf.dtypes.cast(1-domain_gt, tf.bool)) #we only want to keep target which is the second part of the predictions with domain_gt=0
    if mask is not None:
        predictions_targetmatched= tf.boolean_mask(predictions_target, tf.dtypes.cast(mask,tf.bool)) #mask that selects the paired / matched samples with labels    
        if normalize_weights:
            return tf.reduce_sum(mask)*(predictions_targetmatched)/tf.reduce_sum(predictions_targetmatched) #if the GAN classifier is more confused, it's going to predict higher values for target (whereas target label is 0)
    else: #no mask for the case where we want to select all the target samples
        predictions_targetmatched= tf.boolean_mask(predictions_target, tf.dtypes.cast(mask,tf.bool)) #no mask
        if normalize_weights:
            return tf.shape(predictions_targetmatched)[0]*predictions_targetmatched/tf.reduce_sum(predictions_targetmatched)
    return predictions_targetmatched #if no normalization, we don't scale the weights to have a sum equal to the number of samples

def kl_loss(true, pred, active_bool=None):
    kl = tf.keras.losses.KLDivergence(reduction=tf.compat.v1.losses.Reduction.NONE, name='kl')
    loss= kl(true,pred)
    if active_bool is not None:
        loss=tf.math.multiply(active_bool, loss)
        total_count2= tf.to_float(tf.reduce_sum(tf.dtypes.cast(active_bool, tf.int32))) #V2 adaptation Elisabeth 11/08/'20
#    meanres = tf.cond(tf.equal(total_count2, tf.constant(0, dtype=tf.float32)), lambda: tf.constant(0.0, tf.float32), lambda: tf.reduce_sum(loss)/total_count2)
    return loss, tf.reduce_sum((loss)) 