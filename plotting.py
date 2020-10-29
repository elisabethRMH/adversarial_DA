#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:27:32 2020

@author: ehereman
"""
import os
import matplotlib.pyplot as plt
import numpy as np

#direct1='/volume1/scratch/ehereman/results_adversarialDA/try4/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/try4/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/try4/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/try4/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try1/SEMISUP0.99unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/SEMISUP0.99unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34_noNorm/SEMISUP0.0unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try5_newNorm/SEMISUP0.99unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try6/SEMISUP0.99unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try8_newNorm/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try8_newNorm/SEMISUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try8_newNorm/SEMISUP0.9unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try9_newNorm/SEMISUP0.99unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/SEMISUP0.99unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try10_newNormImp2/SEMISUP0.0unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try7/SEMISUP0.99unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try11_newNorm2/SEMISUP0.99unlabeled/'
#
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try12_newNorm2_L2reg/SEMISUP0.99unlabeled/'


direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try13_newNorm2_L2reg_minlayer/SEMISUP0.99unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try14_newNorm2_L2reg/SEMISUP0.99unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/FULLYSUP0.99unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try15_newNorm2_L2reg_minlayer/SEMISUP0.99unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try16_newNorm2_L2regexcDA_minlayer/SEMISUP0.7unlabeled/'
#
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/C34vsF34try17_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'


direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try1_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try2_L2reg_minlayer/SEMISUP0.7unlabeled/'



#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try4_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'


direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try5_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_2/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_newNorm2_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer/SEMISUP0.7unlabeled/'
#
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_2/SEMISUP0.7unlabeled/'
#
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try6_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_2/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try7_newNorm2_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'


direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsF34try8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_2/SEMISUP0.7unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry1_newNorm2_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.7unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.7unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry2_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
##direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
##direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_newNorm2_L2reg_minlayer_eogeval_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_onlyFS/FULLYSUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_onlyFS_2/FULLYSUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry3_onlyFS_3/FULLYSUP0.8unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer/SEMISUP0.7unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_2/SEMISUP0.7unlabeled/'
#
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry4_newNorm2_L2reg_minlayer_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry5_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry6_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled_higherdomainversion/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry7_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry8_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry9_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry10_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
##
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry11_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry12_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry13_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry14_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry15_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
####
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_4/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_5/SEMISUP0.7unlabeled/'
####
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry16_newNorm2_L2reg_minlayer_equalbatches_6/SEMISUP0.7unlabeled/'

direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.8unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.9unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/FULLYSUP0.7unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_2/SEMISUP0.7unlabeled/'
####
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry17_newNorm2_L2reg_minlayer_equalbatches_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100/FULLYSUP0.95unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100/SEMISUP0.95unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100/FULLYSUP0.97unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100/SEMISUP0.97unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_2/FULLYSUP0.95unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_2/SEMISUP0.95unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_2/FULLYSUP0.97unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_2/SEMISUP0.97unlabeled/'
####
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_3/FULLYSUP0.95unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_3/SEMISUP0.95unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_3/FULLYSUP0.97unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry18_batch100_3/SEMISUP0.97unlabeled/'


#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput/SEMISUP0.7unlabeled/'
###
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_2/SEMISUP0.7unlabeled/'
####
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/FULLYSUP0.8unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/SEMISUP0.8unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/FULLYSUP0.9unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/SEMISUP0.9unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/FULLYSUP0.7unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/V2_C34vsEOGtry19_classifierinput_3/SEMISUP0.7unlabeled/'

#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn/FULLYSUP0.0unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_2/SEMISUP0.0unlabeled/'
##direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_2/FULLYSUP0.0unlabeled/'
direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch/SEMISUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch/FULLYSUP0.0unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_LRup/SEMISUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_LRup/FULLYSUP0.0unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_LRup_regdown/SEMISUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_LRup_regdown/FULLYSUP0.0unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum/SEMISUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum/FULLYSUP0.0unlabeled/'
#direct2='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_n1Huy/SEMISUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_n1Huy/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_2/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_3/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_4/FULLYSUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum/FULLYSUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_2/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_3/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_1ch_losssum_4/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineF34_e2earnn_1ch_losssum_5/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineF34_e2earnn_1ch_losssum_6/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineF34_e2earnn_1ch_losssum_7/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineF34_e2earnn_1ch_losssum_8/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_n1Huy/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_2/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_3/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baseline_e2earnn_3ch_losssum_4/FULLYSUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_2/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_3/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_5/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_6/FULLYSUP0.0unlabeled/'
#direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_7/FULLYSUP0.0unlabeled/'
direct1='/volume1/scratch/ehereman/results_adversarialDA/baselineEOG_e2earnn_1ch_losssum_only5pat/FULLYSUP0.0unlabeled/'


file= 'eval_result_log.txt'
file= 'test_result_log.txt'
acc_list=[]

for direct in [direct1]:
    file1= open(os.path.join(direct, file), "r")
    endfile = False
    acc=[]
    while not endfile:
        a=(file1.readline())
        a=a.split()
        if len(a)==0:
            endfile=True
        else:
            a= float(a[-1])
            acc.append(a)
            
    #acc = np.convolve(acc, np.ones((20,))/20, mode='valid')

    acc_list.append(np.array(acc))
#fig=plt.figure()

#labels=['shared weights', 'different weights', 'class weighted loss'] #
labels = ['Fully supervised','Semi-supervised']
colors=['k','r','c','m','b','g']#'k',
#plt.xlabel('Train step (10^3)')
plt.figure()
plt.ylabel('Validation accuracy')
for i in range(len(acc_list)):
    accs= np.array(acc_list[i])
    plt.plot(np.array(range(len(accs)))*500, accs, label= labels[i],c=colors[i], alpha=.4) #*0.5 , linestyle='--'
plt.legend()
plt.title('Normalization method 3, 70% unlabelled, regularized & only 1 FC layer')
#plt.savefig('regr_ss1-3')
plt.show()



# =============================================================================
# #acc_smooth = np.convolve(acc, np.ones((20,))/20, mode='valid')
# labels=['shared weights', 'different weights']
# plt.xlabel('Train step (10^3)')
# plt.ylabel('Validation accuracy')
# for i in range(len(acc_list)):
#     accs= np.array(acc_list[i])c
#     plt.plot(np.array(range(len(accs)))*.5, accs, label= labels[i])
# plt.legend()
# 
# plt.savefig('selfsup_learning')
# plt.show()
# 
# 
# =============================================================================
