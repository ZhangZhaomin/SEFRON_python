# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:46:35 2019

@author: ZhangZhaomin
"""

import numpy as np
import pandas as pd
import parameter as par
from copy import deepcopy
from matplotlib import pyplot
#from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

def loda_data():
    train=np.array(pd.read_csv('iris_train_1.csv', header=None).values)
    test=np.array(pd.read_csv('iris_test_1.csv', header=None).values)
    train_data=train[:,0:par.feature]
    train_label=train[:,par.feature]
    test_data=test[:,0:par.feature]
    test_label=test[:,par.feature]
    return train_data,train_label,test_data,test_label


def spiking_genersate(train_data):
    mu=par.mu
    sigma=par.sigma
    RF=par.RF
    feature=par.feature
    spike_time=np.zeros((int(train_data.shape[0]),par.feature*par.RF))
    for num in range(train_data.shape[0]):
        fire_value=np.zeros(RF)
        fire_strength=np.zeros((feature,RF))
        for t in range(par.feature):
            for i in range(par.RF):
                fire_value[i]=np.exp(-np.power((train_data[num,t]-mu[i]),2)/(2*np.power(sigma,2)))
            fire_strength[t,:]=fire_value
        spike_time[num,:]=(np.round(par.T_pre*(1-fire_strength))+1).reshape((1,par.feature*par.RF))
        pyplot.plot(spike_time.T,'.k')
    return spike_time

def weights_thershold_init():
    weights=np.zeros((par.RF*par.feature,par.T_pre+1,par.class_num))
    theta=np.zeros(par.class_num)
    return weights,theta

def weights_thershold_reload(train_data,train_label,weights,theta,fire_time):
    Output_size=0
    for i in range(train_data.shape[0]):
        if (theta[int(train_label[i]-1)]==0):
                Output_size = Output_size + 1
                weights[:, :, int(train_label[i]-1)] = np.multiply((gaussian_function(par.T_train, fire_time[i,:].transpose())).transpose(), do_uki((par.TID- fire_time[i,:]))[:, np.newaxis])  #timing-vary weigts,weight initial
                theta[int(train_label[i]-1)] = (np.matmul(do_uki((par.TID- fire_time[i,:]))[np.newaxis, :], LIF_e(par.TID - fire_time[i,:].transpose())[:, np.newaxis])).squeeze()     #theta initial
        if (Output_size == par.class_num):   
            break
    return weights,theta

def weights_thershold_update(train_data,train_label,weights,theta,fire_time):
    for i in range(train_data.shape[0]):
        ta,V=PostFiringTime(weights,theta,fire_time[i,:])
        unfire_class = np.array(list(set(np.arange(par.class_num)) - set([int(train_label[i]-1)]))) 
        ta_fire = ta[int(train_label[i]-1)]
        ta_other = np.amin(ta[unfire_class])
        reference_time = deepcopy(ta)
        if (ta_other < ta_fire+par.T_divide):
            if (ta_fire > par.TID-1):
#                print(i)
                reference_time[int(train_label[i]-1)] = par.TID - 1
            trf_mc = min(par.TPost-1, ta_fire+par.T_divide)
            Wrng_class = np.where(ta[unfire_class] < ta_fire+par.T_divide)[0]
            reference_time[unfire_class[Wrng_class]] = trf_mc
            weights=caculate_update(ta,reference_time,fire_time[i,:],theta,weights)
    return weights
        
def caculate_update(ta,reference_time,fire_time,theta,weights):
    deltat_a = (ta[np.newaxis, :]+1) - fire_time[:, np.newaxis]
    uki_ta=do_uki(deltat_a)
    deltat_d=(reference_time[np.newaxis, :]+1) - fire_time[:, np.newaxis]
    uki_td=do_uki(deltat_d)
    gama_ta = theta / np.sum(uki_ta * LIF_e((ta[np.newaxis, :]+1) - fire_time[:, np.newaxis]), axis=0)
    gama_td = theta / np.sum(uki_td * LIF_e((reference_time[np.newaxis, :]+1) - fire_time[:, np.newaxis]), axis=0)   
    delta_W = np.multiply(uki_td, (gama_td - gama_ta)[np.newaxis, :])
    gki = np.multiply(gaussian_function(np.arange(par.T_pre+1), np.tile(fire_time, (1, par.class_num)).squeeze()).transpose(), np.ravel(delta_W, order='F')[:, np.newaxis])
    weights_new=weights+par.lamada * np.transpose(np.reshape(gki.transpose(), (int(par.T_pre+1), np.array(par.feature*par.RF, dtype=np.int32), par.class_num), order='F'), (1, 0, 2))
    weights_new[weights_new==-np.inf] = np.inf
    weights_new[np.isnan(weights_new)] = np.inf
    return weights_new
    
    
    
def PostFiringTime(weights,theta,fire_time):    
    W_sample = weights[np.tile(np.arange(par.feature*par.RF), (1, par.class_num)),
                             np.array(np.tile(fire_time, (1, par.class_num)), dtype=np.int32) - 1,
                             np.ravel(np.tile(np.arange(par.class_num), (par.feature*par.RF, 1)), order='F')]  
    wh = np.reshape(W_sample, (par.feature*par.RF, par.class_num), order='F')
    V = np.matmul(wh.transpose(), LIF_e(par.T_post -fire_time[:, np.newaxis]))
    firing = (V>theta[:, np.newaxis])    
    which_fire = np.argmax(firing.transpose(), axis=0)
    which_fire[which_fire==0] = par.TPost - 1
    ta = which_fire
    return ta, V

def gaussian_function(A,B):
    result = []
    for i in range(A.shape[0]):
           result.append(np.exp((-np.power(A[i]-B, 2)) / (2*(55**2))))
    return np.array(result)


def do_uki(deltat):
    stdp = np.exp(-np.abs(deltat)/(par.stdp*100))    #Stdp equation

    g = deepcopy(deltat)    # delta t: td-ta
    g[g>0] = 1   
    g[g<0] = 0   
    g = g*stdp    

    if (isinstance(np.sum(g, axis=0), np.ndarray)):   
        pos_weight = g/((np.sum(g, axis=0))[np.newaxis, :])
    else:
        pos_weight = g/np.sum(g)

    temp = np.isnan(pos_weight)
    pos_weight[temp]=0

    g= deepcopy(deltat)
    g[g>0] = 0
    g[g<0] = 1
    g = g * stdp

    if (isinstance(np.sum(g, axis=0), np.ndarray)):
        neg_weight = -1 * (g/((np.sum(g, axis=0))[np.newaxis, :]))
    else:
        neg_weight = -1 * (g/np.sum(g))

    temp = np.isnan(neg_weight)
    neg_weight[temp] = 0

    uki = pos_weight + neg_weight

    return uki
    
def LIF_e(deltat):
    x = deltat/(par.tau*100)
    LIF_norm = x * np.exp(1-x)   
    LIF_norm[LIF_norm<0]=0

    return LIF_norm    
    
def validate(Spike_data, clas,weights,theta):
    Output_class = np.zeros(clas.shape, dtype=np.int32) -1

    for j in range(Spike_data.shape[0]):
        tc,Fire_vk=PostFiringTime(weights,theta,Spike_data[j,:])
        
        val = np.where(np.amin(tc)==tc)[0]
        val = val[np.newaxis, :]
        vd1 = np.amin(tc)

        if (vd1 != par.TPost-1 and val.shape[0]==1):
            Output_class[j] = int(np.argmin(tc))
        elif (val.shape[0] != 1 and vd1 != par.TPost-1):
            reqd_indices = np.array([[x, y] for x in np.ravel(val, order='F') for y in np.arange(vd1-1, vd1+1)])
            fire_precision = Fire_vk[reqd_indices[:, 0], reqd_indices[:, 1]]
            fire_precision = fire_precision.reshape((np.prod(val.shape), int(fire_precision.shape/np.prod(val.shape))))
            ds = fire_precision - theta[val]
            temp_ds = (np.zeros(val.shape) - ds[:, 1]) / (ds[:, 2] - ds[:, 1])
            f = np.argmin(temp_ds)
            Output_class[j] = val[f]     
    clas = np.array(clas, dtype=np.int) - 1
    Output_class = np.array (Output_class, dtype=np.int32)
    correct = np.where(Output_class == clas)[0]
    accuracy = correct.shape[0] / Spike_data.shape[0] * 100
    return accuracy

def main():
    train_data,train_label,test_data,test_label=loda_data()
    ###spikes generate
    fire_time=spiking_genersate(train_data)
    test_spike=spiking_genersate(test_data)
    Train_accuracy = np.zeros(par.epoch)
    Test_accuracy = np.zeros(par.epoch)
    ###weights/threshold initilize
    weights,theta=weights_thershold_init()
    ### base input_spikes initial weights/thershold
    weights,theta=weights_thershold_reload(train_data,train_label,weights,theta,fire_time)
    ### base input_dara update weights
    for i in range (par.epoch):
        weights=weights_thershold_update(train_data,train_label,weights,theta,fire_time)
        if i %25==0:
            np.savetxt('weights'+str(i)+'.csv',weights[:,i,:])
        Train_accuracy[i]=validate(fire_time, train_label,weights,theta)
        Test_accuracy[i]=validate(test_spike, test_label,weights,theta)
        print('epoch='+str(i)+':'+'Train_accurcay='+str(Train_accuracy[i])+','+'Test_accuracy='+str(Test_accuracy[i]))
    pyplot.figure(2)
    pyplot.plot(Train_accuracy,'r',label='Training')
    pyplot.plot(Test_accuracy,'b',label='Inference')
    pyplot.title('Accuracy of training&infercence(%)')
    pyplot.xlabel('epoch')
    pyplot.ylabel('accurcay(%)')
    pyplot.legend(loc='lower right')
    
    pyplot.figure(3)
    return

if __name__ == '__main__':
    main()
