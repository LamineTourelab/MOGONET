#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:39:37 2022

@author: ltoure
"""
import os
import pandas as pd 
import numpy as np 
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from train_test import train_test, prepare_trte_data, train_epoch, test_epoch, gen_trte_adj_mat
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
import copy

#### Define device ###
cuda = True if torch.cuda.is_available() else False

#####################################
#           Load Data               #
#####################################

rnaseq = pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/Data/Rnaseq", sep=",", index_col=0)
traitData = pd.read_csv("//home/ldap/ltoure/multiomics/Multiomics/Input/Data/Response", index_col=0)
DataExome= pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/Data/Exome", sep=",", index_col=0)
TestIndex100 = pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/TestIndex100Split", sep=" ")

#############################################
#          Data transformation              #
#############################################


test_i = TestIndex100
test_i = test_i - 1

 #with open('xgboost_opt_params', 'rb') as f:
  #  params = pickle.load(f)
X= rnaseq
y= DataExome
labels=traitData

data_folder = 'ROSMAP'
view_list = [1,2]
num_epoch_pretrain = 500
num_epoch = 1
lr_e_pretrain = 1e-3
lr_e = 5e-4
lr_c = 1e-3
    
if data_folder == 'ROSMAP':
        num_class = 2
if data_folder == 'BRCA':
        num_class = 5
        
#all_imp = {} 
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

    labels_tr = labels_tr.replace("R",1.0000+00)
    labels_tr = labels_tr.replace("NR",0.0000+00)
    labels_te = labels_te.replace("NR",0.0000+00)
    labels_te = labels_te.replace("R",1.0000+00)

    feature_name_X_tr=X_tr.columns
    feature_name_y=y_tr.columns

    X_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/1_tr.csv", header=False, index=False)
    X_te.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/1_te.csv", header=False, index=False)
    y_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/2_tr.csv", header=False, index=False)
    y_te.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/2_te.csv", header=False, index=False)
  #  X_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/3_tr.csv", header=False, index=False)
   # X_te.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/3_te.csv", header=False, index=False)
    labels_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/labels_tr.csv", header=False, index=False)
    labels_te.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/labels_te.csv", header=False, index=False)
    feature_name_X_tr=pd.DataFrame(feature_name_X_tr) 
    feature_name_X_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/1_featname.csv", header=False, index=False)
    feature_name_y=pd.DataFrame(feature_name_y) 
    feature_name_y.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/2_featname.csv", header=False, index=False)
   # feature_name_X_tr_tr=pd.DataFrame(feature_name_X_tr) 
    #feature_name_X_tr.to_csv("/home/ldap/ltoure/MOGONET/ROSMAP/3_featname.csv", header=False, index=False)
    
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
       # print("\nPretrain GCNs: Epoch {:d}".format(epoch))
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
                auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])
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

            data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            featname_list = []
            for v in view_list:
                df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
                featname_list.append(df.values.flatten())
            
            dim_list = [x.shape[1] for x in data_tr_list]
            model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
            for m in model_dict:
                if cuda:
                    model_dict[m].cuda()
            #model_dict = load_model_dict(model_folder, model_dict)
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                   
            
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
                    data_trte_list[i][:,j] = feat_trte.clone()
                feat_imp_list.append(pd.DataFrame(data=feat_imp))
    
featimp_list_list = []
featimp_list_list.append(copy.deepcopy(feat_imp_list))
#def summarize_imp_feat(featimp_list_list, topn=30):
num_rep = len(featimp_list_list)
num_view = len(featimp_list_list[0])
df_tmp_list = []
for v in range(num_view):
    df_tmp = copy.deepcopy(featimp_list_list[0][v])
    df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
    df_tmp_list.append(df_tmp.copy(deep=True))
df_featimp = pd.concat(df_tmp_list).copy(deep=True)
for r in range(1,num_rep):
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[r][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True) 
df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
df_featimp_top = df_featimp_top.reset_index()
df_featimp_top = df_featimp_top.sort_values(by='imp',ascending=False)
df_featimp_top = df_featimp_top.iloc[:30]
print('{:}\t{:}'.format('Rank','Feature name'))
for i in range(len(df_featimp_top)):
    print('{:}\t{:}'.format(i+1,df_featimp_top.iloc[i]['feat_name']))
        
#####################################
#           Performance             #
#####################################

feat_imp=sns.barplot(data=df_featimp_top, x="imp", y="feat_name")
fig = feat_imp.get_figure()
fig.savefig('Feature_imp_MOGONET.png')

df_featimp_top.to_csv("/home/ldap/ltoure/MOGONET/Featimp.csv")


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

    
           
    
    
 



















