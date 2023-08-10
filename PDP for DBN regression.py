# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:32:00 2023

@author: pc07
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dbn.tensorflow import SupervisedDBNRegression
from sklearn.inspection import partial_dependence


def data_prerocessing (Input,Output):
    #Input = (Input - np.mean(Input,axis = 0))/np.std(Input,axis = 0)
    Input = Input/np.std(Input,axis = 0)
    (qt,Removal,finalPH,LeachingMg) = (Output[:,0],Output[:,1],Output[:,2],Output[:,3])
    
    fixpoint = dict(qt =  2.03, 
                    Removal = np.max(Removal)- np.min(Removal),
                    finalPH = np.max(finalPH)- np.min(finalPH),
                    LeachingMg = 1.706)
    
    qt_processed = np.emath.logn(fixpoint['qt'],qt)
    Removal_processed = Removal/fixpoint['Removal']
    finalPH_processed = finalPH/fixpoint['finalPH']
    LeachingMg_processed = np.emath.logn(fixpoint['LeachingMg'],LeachingMg+1)    
    Output = np.stack((qt_processed,Removal_processed,finalPH_processed,LeachingMg_processed),axis = 1)
    
    return (Input,Output,fixpoint)

def cal_result(prediction,fix_point):
    
    qt_result = np.power(fix_point['qt'],prediction[:,0])
    Removal_result = prediction[:,1] * fix_point['Removal']
    finalPH_result = prediction[:,2] * fix_point['finalPH']
    LeachingMg_result = np.power(fix_point['LeachingMg'],prediction[:,3])-1
    result = np.stack((qt_result,Removal_result,finalPH_result,LeachingMg_result),axis = 1)
    
    return result

#Loading data
df= pd.read_csv('Data-for-P-MgO.csv')
data = pd.DataFrame.to_numpy(df)   
print(data.shape)

#Data preprocessing
#input of model
X = data[:,0:15]
#output of model
Y = data[:,15:19]
X_,Y_ ,fix_point= data_prerocessing(X,Y)
print(fix_point)

X_,Y_ ,fix_point= data_prerocessing(X,Y)



# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, random_state=1)

# Data scaling
#min_max_scaler = MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)

# Training
#regressor = SupervisedDBNRegression(hidden_layers_structure=[64,16,16,8,4],
                                    #learning_rate_rbm=0.001,
                                    #learning_rate=0.05,
                                    #n_epochs_rbm=100,
                                    #n_iter_backprop=200000,
                                    #batch_size=512,
                                    #activation_function='relu')
#regressor.fit(X_train, Y_train)
#regressor._fine_tuning(X_train, Y_train)

# Save the model
#regressor.save('best_model.h5')

# Load model
regressor = SupervisedDBNRegression.load('./Model/DBN/DBN_[64,32,16,8].h5')
#regressor = SupervisedDBNRegression.load('./Model/DBN/DBN_[64,16,16,8,4].h5')

feat_name = df.columns.values
std = np.std(X,axis = 0)

# Iterate through each input feature
for i, feature_index in enumerate(range(15)):
    # Create the Partial Dependence plot
    display = partial_dependence(regressor, X_train, features=[feature_index])
    
    feat = np.matrix.transpose(display.get('values')[0]*std[i])
    target = cal_result(np.matrix.transpose(display.get('average')) , fix_point)
    
    save_data = {}  
    
    save_data[feat_name[i]] = feat
    save_data[feat_name[15]] = target[:,0]
    save_data[feat_name[16]] = target[:,1]
    save_data[feat_name[17]] = target[:,2]
    save_data[feat_name[18]] = target[:,3]
    
    data = pd.DataFrame(data = save_data)
    data.to_csv(feat_name[i]+ ' _ PartialDependencePlot_DBNv1.csv',index = False)