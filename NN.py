# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 00:06:32 2019

@author: Guangyu
"""



import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
import struct
import scipy
from scipy import stats
import scipy.io as sio

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit
import numpy as np
import scipy.io as sio  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus 
from numpy import ones
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from keras.models import load_model
from sklearn.externals import joblib 

model = load_model('AE_CNN_same_with_NN.h5')
intermediate_layer_model = load_model('intermediate_layer_model_same_with_NN.h5')
autoencoder = load_model('AE_same_with_NN.h5')



data1 = sio.loadmat('XTrain_same_with_NN_4_1_1_62432.mat')
data1 = data1.get('XTrain')
data1 = np.float32(data1)
data1 = data1.transpose((3,0,1,2))
data1 = np.squeeze(data1)


#data1 = sio.loadmat('XTrain_same_with_NN_4_624320.mat')
#data1 = data1.get('x')

data2 = sio.loadmat('XTest_same_with_NN_4_1_1_312160.mat')
data2 = data2.get('XTest')
data2 = np.float32(data2)
data2 = data2.transpose((3,0,1,2))
data2 = np.squeeze(data2)

ytrain = sio.loadmat('YTrain_same_with_NN_4_1_1_62432.mat')
data_train_label = ytrain.get('YTrain')
ytrain1 = keras.utils.to_categorical(data_train_label,11)
ytrain1 = np.delete(ytrain1,0,axis=1)

#ytrain = sio.loadmat('YTrain_same_with_NN_4_624320.mat')
#data_train_label = ytrain.get('YTrain')
#ytrain1 = keras.utils.to_categorical(data_train_label,11)
#ytrain1 = np.delete(ytrain1,0,axis=1)

ytest = sio.loadmat('YTest_same_with_NN_4_1_1_312160.mat')
data_test_label = ytest.get('YTest')
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i
    ytest_target1.append(temp)
    
    
############### For the top classifier, data:19/01/2020
    
data1 = sio.loadmat('XTrain_for_top_classifier.mat')
data1 = data1.get('XTrain')
data1 = np.float32(data1)
data1 = data1.transpose((1,0))

data2 = sio.loadmat('XTest_for_top_classifier.mat')
data2 = data2.get('XTest')
data2 = np.float32(data2)
data2 = data2.transpose((1,0))

ytrain = sio.loadmat('YTrain_for_top_classifier.mat')
data_train_label = ytrain.get('YTrain')

ytest = sio.loadmat('YTest_for_top_classifier.mat')
data_test_label = ytest.get('YTest')

ytest_target = sio.loadmat('target_test_3_class_19_01_2020.mat')
ytest_target = ytest_target.get('target_test')
ytest_target1 = []
for i in ytest_target:
    temp = i
    ytest_target1.append(temp)
    

data_train_label = np.ravel(data_train_label)
clf_rf = RandomForestClassifier()
clf_rf.fit(data1, data_train_label)
y_pred_rf = clf_rf.predict(data2)
acc_rf = accuracy_score(data_test_label, y_pred_rf)
print(acc_rf)
acc_rf_train = clf_rf.score(data1, data_train_label)
print(acc_rf_train)

#0.9116831112250128
#0.9955791901588928

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_rf[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_rf = a/len(ytest_target1)  
print(final_accuracy_rf)   
#0.9625

clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(data1, data_train_label)
y_pred_decisiontree = clf_dt.predict(data2)
score=clf_dt.score(data2,data_test_label)
print("%f"%score)
score_train=clf_dt.score(data1,data_train_label)
print("%f"%score_train)
#0.898680
#1.000000

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_decisiontree[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_decision_tree = a/len(ytest_target1)  
print(final_accuracy_decision_tree)

##0.975
############### For the top classifier



model = Sequential()
model.add(Dense(units=200,activation='relu', input_dim=4))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='sgd', loss='mean_squared_error',
                    metrics = ['accuracy'])
model.fit(data1, ytrain1,
          batch_size=128,
          epochs=100, 
          verbose=1,
          shuffle=True)

from neupy import algorithms
lmnet = algorithms.LevenbergMarquardt((4, 10, 10),show_epoch=1)
lmnet.train(data1, ytrain1)

###################################noise data for AE+CNN
data2 = sio.loadmat('XTest_same_with_NN_4_1_1_312160.mat')
data2 = data2.get('XTest')
data2 = data2.transpose((3,0,1,2))
########################################noise data for others
data1 = np.reshape(data1,(62432,4))
data2 = np.reshape(data2,(312160,4))

################################## Note the varient 
##################################rf1
data_train_label = np.ravel(data_train_label)
clf_rf = RandomForestClassifier()
clf_rf.fit(data1, data_train_label)
y_pred_rf = clf_rf.predict(data2)
acc_rf = accuracy_score(data_test_label, y_pred_rf)
print(acc_rf)
acc_rf_train = clf_rf.score(data1, data_train_label)
print(acc_rf_train)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_rf[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_rf = a/len(ytest_target1)  
print(final_accuracy_rf)   
#0.7987346232701179
#0.88125

#0.8300198616094311
#0.93125(all of the data)

#0.2984783444387494
#0.35

################################svm
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf_svm = SVC(kernel='rbf')  
clf_svm.fit(data1, y_train)
y_pred_svm = clf_svm.predict(data2)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(acc_svm)
acc_svm_train = clf_svm.score(data1, y_train)
print(acc_svm_train)

y_pred_svm = np.zeros((312160,1))
for i in range(30000,len(data2)):
    a = data2[i].reshape(-1,1)
    y_pred_svm[i] = clf_svm.predict(a.transpose())

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_svm[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_svm = a/len(ytest_target1)  
print(final_accuracy_svm)   
#0.778517426960533
#0.85
from sklearn.externals import joblib 
joblib.dump(clf_svm, 'save_4_312160/clf_rf.pkl')
#clf_svm = joblib.load('save_4_312160/clf_rf.pkl')
##################################dt
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(data1, data_train_label)
y_pred_decisiontree = clf_dt.predict(data2)
score=clf_dt.score(data2,data_test_label)
print("%f"%score)
score_train=clf_dt.score(data1,data_train_label)
print("%f"%score_train)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_decisiontree[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_decision_tree = a/len(ytest_target1)  
print(final_accuracy_decision_tree)
# 0.817299
# 0.9375     

#0.836542
#0.975

#0.458217
#0.83125

dot_data = tree.export_graphviz(clf_dt, out_file=None, 
                     filled=True, rounded=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())
temp=np.array(y_test2)
##################################knn
clf_knn = KNeighborsClassifier(n_neighbors=3) #or 5
clf_knn.fit(data1, data_train_label)
y_pred_knn = clf_knn.predict(data2)
acc_knn = accuracy_score(data_test_label, y_pred_knn)
print(acc_knn)
acc_knn_train= clf_knn.score(data1, data_train_label)
print(acc_knn_train)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_knn[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_knn = a/len(ytest_target1)  
print(final_accuracy_knn) 
#0.7322558944131214
#0.875 

##################################################nn
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf_nn = MLPClassifier(hidden_layer_sizes=(21,21,21,),verbose=1,activation='logistic')
clf_nn.fit(data1, y_train)
y_pred_nn = clf_nn.predict(data2)
acc_nn = clf_nn.score(data2,y_test)
print(acc_nn)
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_nn[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_nn = a/len(ytest_target1)  
print(final_accuracy_nn) 
#0.47038698103536647
#0.475
####################################################sgd
clf_sgd = SGDClassifier()
clf_sgd.fit(data1, data_train_label)
y_pred_sgd = clf_sgd.predict(data2)
acc_sgd = accuracy_score(data_test_label, y_pred_sgd)
print("stochastic gradient descent accuracy: ",acc_sgd)

##########################################################regression
clf = LogisticRegression()
clf.fit(data1,data_train_label)
lr_test_sc=clf.score(data2,data_test_label)
y_pred_reg = clf.predict(data2)
acc_reg = clf.score(data2,data_test_label)
print("regression: ",lr_test_sc)
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_reg[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_reg = a/len(ytest_target1)  
print(final_accuracy_reg) 
#24.37%
###################################################nb

clf_gnb = GaussianNB()
clf_gnb.fit(data1,y_train)
y_pred_gnb = clf_gnb.predict(data2)
acc_gnb = clf_gnb.score(data2, y_test)
print("nb accuracy: ",acc_gnb)
###################################################AE+CNN
from keras.models import load_model
model = load_model('AE_CNN_same_with_NN.h5')
intermediate_layer_model = load_model('intermediate_layer_model_same_with_NN.h5')
autoencoder = load_model('AE_same_with_NN.h5')


ytest = sio.loadmat('YTest_same_with_NN.mat')
data_test_label = ytest.get('YTest')
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i-1
    ytest_target1.append(temp)

intermediate_output_2 = intermediate_layer_model.predict(data2)
y_test = model.predict(intermediate_output_2)
score = model.evaluate(intermediate_output_2, ytest1, verbose=0)

y_test1 = []

for r in y_test:
    r = r.tolist()
    index = r.index(max(r))
    y_test1.append(index)
#    if i > y_test.shape[0]:
#        break
   
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_test1[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy = a/len(ytest_target1)  
print(final_accuracy)      
