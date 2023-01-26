#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:39:37 2022

@author: ltoure
"""
## This model is the summary of the differents modules of origoinal MOGONET model from the model training to features importants and evaluation of the model. 
import pandas as pd 
import numpy as np 
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import pickle
from sklearn.metrics import roc_curve, auc
import csv
import matplotlib.pyplot as plt
#####################################
#           Load Data               #
#####################################
# The model takes 3 types omics data. So here i duplicate one the inputs.
omics1 = pd.read_csv("path", sep=" ")
label = pd.read_csv("path", sep=" ")
omics2 = pd.read_csv("path", sep=" ")
TestIndex100 = pd.read_csv("path", sep=" ") # To allow training in split data 100 times.

#############################################
#          Data transformation              #
#############################################


test_i = TestIndex100
test_i = test_i - 1

X = omics1
y = omics2
labels=traitData

data_folder = 'ROSMAP'
view_list = [1,2,3]
num_epoch_pretrain = 1000
num_epoch = 1500
lr_e_pretrain = 1e-3
lr_e = 5e-4
lr_c = 1e-3
    
if data_folder == 'ROSMAP':
        num_class = 2 # Binary classification
if data_folder == 'BRCA':
        num_class = 5 # Multiclass classification
        
all_auc = []
all_acc = []
all_bacc = []
all_f1 = []
all_precision = []
all_recall = []
all_se = []
all_sp = []

for i in range(test_i.shape[1]):
    X_tr = X.iloc[np.setdiff1d(np.arange(X.shape[0]), test_i.iloc[:,i]),:]
    X_te = X.iloc[test_i.iloc[:,i],:]
    y_tr = y.iloc[np.setdiff1d(np.arange(y.shape[0]), test_i.iloc[:,i]),:]
    y_te = y.iloc[test_i.iloc[:,i],:]
    labels_tr = labels.iloc[np.setdiff1d(np.arange(y.shape[0]), test_i.iloc[:,i]),:]
    labels_te = labels.iloc[test_i.iloc[:,i],:]

    labels_tr = labels_tr.replace("labels1",1.0000+00)
    labels_tr = labels_tr.replace("labels2",0.0000+00)
    labels_te = labels_te.replace("labels1",0.0000+00)
    labels_te = labels_te.replace("labels2",1.0000+00)

    feature_name_X_tr=X_tr.columns
    feature_name_y=y_tr.columns

    X_tr.to_csv("~/MOGONET/ROSMAP/1_tr.csv", header=False, index=False)
    X_te.to_csv("~/MOGONET/ROSMAP/1_te.csv", header=False, index=False)
    y_tr.to_csv("~/MOGONET/ROSMAP/2_tr.csv", header=False, index=False)
    y_te.to_csv("~/MOGONET/ROSMAP/2_te.csv", header=False, index=False)
    # While I have only two OMICs data, I duplicate the type 1.
    X_tr.to_csv("~/MOGONET/ROSMAP/3_tr.csv", header=False, index=False)
    X_te.to_csv("~/MOGONET/ROSMAP/3_te.csv", header=False, index=False)
    labels_tr.to_csv("~/MOGONET/ROSMAP/labels_tr.csv", header=False, index=False)
    labels_te.to_csv("~/MOGONET/ROSMAP/labels_te.csv", header=False, index=False)
    feature_name_X_tr=pd.DataFrame(feature_name_X_tr) 
    feature_name_X_tr.to_csv("~/MOGONET/ROSMAP/1_featname.csv", header=False, index=False)
    feature_name_y=pd.DataFrame(feature_name_y) 
    feature_name_y.to_csv("~/MOGONET/ROSMAP/2_featname.csv", header=False, index=False)
    feature_name_X_tr_tr=pd.DataFrame(feature_name_X_tr) 
    feature_name_X_tr.to_csv("~/MOGONET/ROSMAP/3_featname.csv", header=False, index=False)
    
    #####################################  
    #           Model train              #
    #####################################
    
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,400,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
                
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                auc=roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                bacc = balanced_accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                precision = precision_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                recall = recall_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                cm = confusion_matrix(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
                specificity = cm[1,1]/(cm[1,0]+cm[1,1])
                all_auc.append(auc)
                all_f1.append(f1)
                all_acc.append(acc)
                all_bacc.append(bacc)
                all_precision.append(precision)
                all_recall.append(recall)
                all_se.append(sensitivity)
                all_sp.append(specificity)
                
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()
            
#####################################
#        Feature Importance         #
#####################################
            
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
        featname_list.append(df.values.flatten())
    feat_imp_list = []
    for i in range(len(featname_list)):
        feat_imp = {"feat_name":featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
    for j in range(dim_list[i]):
        feat_tr = data_tr_list[i][:,j].clone()
        feat_trte = data_trte_list[i][:,j].clone()
        data_tr_list[i][:,j] = 0
        data_trte_list[i][:,j] = 0
        adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        if num_class == 2:
           f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        else:
           f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
        feat_imp['imp'][j] = (f1-f1_tmp)*dim_list[i]
                
        data_tr_list[i][:,j] = feat_tr.clone()
        data_trte_list[i][:,j] = feat_trte.clone()bb
    feat_imp_list.append(pd.DataFrame(data=feat_imp))
        
#####################################
#           Performance             #
#####################################

feat_imp=sns.barplot(data=df_featimp_top, x="imp", y="feat_name")
fig = feat_imp.get_figure()
fig.savefig('Feature_imp_MOGONET.png')

df_featimp_top.to_csv("~/MOGONET/Featimp.csv")


hist_auc = sns.displot(all_auc)
hist_auc.savefig('AUC_MOGONET_100_splits.png')

hist_acc = sns.displot(all_acc)
hist_acc.savefig('Accuracy_MOGONET_100_splits.png')

hist_bacc = sns.displot(all_bacc)
hist_bacc.savefig('BalAccuracy_MOGONET_100_splits.png')

hist_f1 = sns.displot(all_f1)
hist_f1.savefig('F1_MOGONET_100_splits.png')

hist_precision = sns.displot(all_precision)
hist_precision.savefig('Precision_MOGONET_100_splits.png')

hist_recall = sns.displot(all_recall)
hist_recall.savefig('Recall_MOGONET_100_splits.png')

hist_se = sns.displot(all_se)
hist_se.savefig('Sensitivity_MOGONET_100_splits.png')

hist_sp = sns.displot(all_sp)
hist_sp.savefig('Specificity_MOGONET_100_splits.png')

mean_perf = {'auc' : np.mean(all_auc),
             'acc' : np.mean(all_acc),
             'bacc' : np.mean(all_bacc),
             'f1' : np.mean(all_f1),
             'prec' : np.mean(all_precision),
             'recall' : np.mean(all_recall),
             'sens' : np.mean(all_se),
             'spec' : np.mean(all_sp)}

mean_perf_out = open("MeanPerf_MOGONET_100_splits", "w")
for key in mean_perf.keys():
   mean_perf_out.write(key + " " + str(mean_perf[key]) + "\n")
mean_perf_out.close() 

    
           
    
    
 



















