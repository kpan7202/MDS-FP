# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:19:39 2017

@author: Kurniawan
"""

import time
import random
import os 
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import sklearn.ensemble as ens
import sklearn.neural_network as net
import sklearn.metrics as met
import sklearn.model_selection as mod
import imblearn.under_sampling  as ius
import imblearn.over_sampling as ios
import imblearn.ensemble as ien
from datetime import datetime
from collections import Counter
from sklearn.decomposition import PCA
from utilities.witmart import WitMartJobs,WitMartUsers

def fit_score(network_train_set, network_test_set, task_train_set, task_test_set, rf, gb):
    scores = np.zeros((5,4))
    # random forest on network data
    rf.fit(network_train_set[:,:-1],network_train_set[:,-1])
    rf_network_pred = rf.predict(network_test_set[:,:-1])
    
    scores[0,0] = met.accuracy_score(network_test_set[:,-1], rf_network_pred)
    scores[0,1] = met.precision_score(network_test_set[:,-1], rf_network_pred)
    scores[0,2] = met.recall_score(network_test_set[:,-1], rf_network_pred)
    scores[0,3] = met.f1_score(network_test_set[:,-1], rf_network_pred)
    
    # random forest on ordinary data
    rf.fit(task_train_set[:,:-1],task_train_set[:,-1])
    rf_task_pred = rf.predict(task_test_set[:,:-1])
    
    scores[1,0] = met.accuracy_score(task_test_set[:,-1], rf_task_pred)
    scores[1,1] = met.precision_score(task_test_set[:,-1], rf_task_pred)
    scores[1,2] = met.recall_score(task_test_set[:,-1], rf_task_pred)
    scores[1,3] = met.f1_score(task_test_set[:,-1], rf_task_pred)
    
    # gradient boosting on network data
    gb.fit(network_train_set[:,:-1],network_train_set[:,-1])
    gb_network_pred = gb.predict(network_test_set[:,:-1])
    
    scores[2,0] = met.accuracy_score(network_test_set[:,-1], gb_network_pred)
    scores[2,1] = met.precision_score(network_test_set[:,-1], gb_network_pred)
    scores[2,2] = met.recall_score(network_test_set[:,-1], gb_network_pred)
    scores[2,3] = met.f1_score(network_test_set[:,-1], gb_network_pred)
    
    # gradient boosting on ordinary data
    gb.fit(task_train_set[:,:-1],task_train_set[:,-1])
    gb_task_pred = gb.predict(task_test_set[:,:-1])
    
    scores[3,0] = met.accuracy_score(task_test_set[:,-1], gb_task_pred)
    scores[3,1] = met.precision_score(task_test_set[:,-1], gb_task_pred)
    scores[3,2] = met.recall_score(task_test_set[:,-1], gb_task_pred)
    scores[3,3] = met.f1_score(task_test_set[:,-1], gb_task_pred)
    
    # ensemble on both data
    # add another prediction on ordinary data with random forest
    rf.fit(task_train_set[:,:-1],task_train_set[:,-1])
    rf_task_pred_2 = rf.predict(task_test_set[:,:-1])
    
    majority_pred = (((rf_network_pred + rf_task_pred + gb_network_pred + gb_task_pred + rf_task_pred_2) / 5) > 0.5).astype(int)
    scores[4,0] = met.accuracy_score(task_test_set[:,-1], majority_pred)
    scores[4,1] = met.precision_score(task_test_set[:,-1], majority_pred)
    scores[4,2] = met.recall_score(task_test_set[:,-1], majority_pred)
    scores[4,3] = met.f1_score(task_test_set[:,-1], majority_pred)
    
    return scores

def sample_fit_test(network_ds,task_ds,label_1_idx, label_0_idx,rf,gb,n_sample = 6000, n_iter = 10,folds = 10, test_size = 0.05):
    print("fit & test models with sample", n_sample, "and", folds, " folds cv")
    scores = np.zeros((2,5,n_iter,4)) # 2-> 0:cv, 1:test; 5-> number of models; 4-> metrics (acc,pre,rec,f1)
    for i in range(n_iter):
        print("Iteration", i+1)
        #split 5% data for test set
        train_1_idx = np.random.choice(label_1_idx, int((1 - test_size) * len(label_1_idx)), False)
        test_1_idx = np.setdiff1d(label_1_idx, train_1_idx, True)
        train_0_idx = np.random.choice(label_0_idx, int((1 - test_size) * len(label_0_idx)), False)
        test_0_idx = np.setdiff1d(label_0_idx, train_0_idx, True)
#        train_data = np.append(train_1_idx, train_0_idx)
        test_data = np.append(test_1_idx, test_0_idx)
        
#        # apply PCA
#        pca = PCA(n_components=0.8)
#        pca.fit(network_ds[train_data,:-1])
#        pc_network_train = np.hstack(( pca.transform(network_ds[train_data,:-1]), np.matrix(network_ds[train_data,-1]).T ))
#        pc_network_test = np.hstack(( pca.transform(network_ds[test_data,:-1]), np.matrix(network_ds[test_data,-1]).T ))
#        pca.fit(task_ds[train_data,:-1])
#        pc_task_train = np.hstack(( pca.transform(task_ds[train_data,:-1]), np.matrix(task_ds[train_data,-1]).T ))
#        pc_task_test = np.hstack(( pca.transform(task_ds[test_data,:-1]), np.matrix(task_ds[test_data,-1]).T ))

        # manual sample
        # under sampling
        sample_0_idx = np.random.choice(train_0_idx, n_sample, False)
        # over_sampling
        sampling_1_idx = np.append(train_1_idx, np.random.choice(train_1_idx, n_sample - len(train_1_idx), True))
        combined_index = np.append(sample_0_idx,sampling_1_idx)
        sampled_network = network_ds[combined_index,:]
        sampled_task = task_ds[combined_index,:]
        
#        sample_0_idx = np.random.choice(np.arange(len(train_1_idx),len(train_data)), n_sample, False)
#        # over_sampling
#        sampling_1_idx = np.append(np.arange(len(train_1_idx)), np.random.choice(np.arange(len(train_1_idx)), n_sample - len(train_1_idx), True))
#        combined_index = np.append(sample_0_idx,sampling_1_idx)
#        sampled_network = pc_network_train[combined_index,:]
#        sampled_task = pc_task_train[combined_index,:]

        #Split the data based on number of folds        
        kf = mod.KFold(folds, shuffle=True)
        cv_scores = np.zeros((5,folds,4))
        j = 0
        for train,test in kf.split(sampled_task):
            #Split the data into train and validation set
            network_train_set = sampled_network[train,:]
            task_train_set = sampled_task[train,:]
            network_test_set = sampled_network[test,:]
            task_test_set = sampled_task[test,:]
            
            cv_scores[:,j,:] = fit_score(network_train_set, network_test_set, task_train_set, task_test_set, rf, gb)
        
            j += 1
            print("Fold", j)
        
        # average of cv scores of iteration i
        scores[0,:,i,:] = np.mean(cv_scores,axis=1)
        # test score on sampled dataset of iteration i
        scores[1,:,i,:] = fit_score(sampled_network, network_ds[test_data,:], sampled_task, task_ds[test_data,:], rf, gb)
#        scores[1,:,i,:] = fit_score(sampled_network, pc_network_test, sampled_task, pc_task_test, rf, gb)
        
    return scores

# cross validation for imbalanced data with sampling method, assuming the column label is the last column of the dataset
def cv_sampling_imbalanced(label_1_idx, label_0_idx, ds, classifier, sampler, folds = 10, sample_test = False):
    # apply cross validation
    train_size = 1 - 1 / folds
    scores = np.zeros((folds,4))
    for i in range(folds):
        # randomly select and split the index of data for training and test set
        train_1_idx = np.random.choice(label_1_idx, int(train_size * len(label_1_idx)), False)
        test_1_idx = np.setdiff1d(label_1_idx, train_1_idx, True)
        train_0_idx = np.random.choice(label_0_idx, int(train_size * len(label_0_idx)), False)
        test_0_idx = np.setdiff1d(label_0_idx, train_0_idx, True)
        train_data = np.append(train_1_idx, train_0_idx)
        test_data = np.append(test_1_idx, test_0_idx)
        
        if sample_test:
            if 'ensemble' in str(sampler.__class__): # ensemble based sampling generates more than 1 sample
                Xs, Ys = sampler.fit_sample(ds[train_data,:-1], ds[train_data,-1].astype(int))
                Xt, Yt = sampler.fit_sample(ds[test_data,:-1], ds[test_data,-1].astype(int))
                
                sampling_scores = np.zeros((Yt.shape[0],4))
                for j in range(Yt.shape[0]):
                    classifier.fit(Xs[j],Ys[j])
                    Ypred = classifier.predict(Xt[j])
                    
                    sampling_scores[j,0] = met.accuracy_score(Yt[j], Ypred)
                    sampling_scores[j,1] = met.precision_score(Yt[j], Ypred)
                    sampling_scores[j,2] = met.recall_score(Yt[j], Ypred)
                    sampling_scores[j,3] = met.f1_score(Yt[j], Ypred)
                scores[i] = np.mean(sampling_scores,axis = 0)
            else:
                Xs, Ys = sampler.fit_sample(ds[train_data,:-1],ds[train_data,-1])
                Xt, Yt = sampler.fit_sample(ds[test_data,:-1],ds[test_data,-1])
                classifier.fit(Xs,Ys)
                test_result = classifier.predict(Xt)
 
                scores[i,0] = met.accuracy_score(Yt, test_result)
                scores[i,1] = met.precision_score(Yt, test_result)
                scores[i,2] = met.recall_score(Yt, test_result)
                scores[i,3] = met.f1_score(Yt, test_result)
    
        else:
            if 'ensemble' in str(sampler.__class__): # ensemble based sampling generates more than 1 sample
                Xs, Ys = sampler.fit_sample(ds[train_data,:-1], ds[train_data,-1].astype(int))
                Ypred = np.zeros((Ys.shape[0],len(test_data)),int)
                for j in range(Ys.shape[0]):
                    classifier.fit(Xs[j],Ys[j])
                    Ypred[j] = classifier.predict(ds[test_data,:-1])
                # use majority vote to determine the label
                test_result = (np.sum(Ypred,0) > (Ys.shape[0] / 2)).astype(int)
            else:
                Xs, Ys = sampler.fit_sample(ds[train_data,:-1],ds[train_data,-1])
                classifier.fit(Xs,Ys)
                test_result = classifier.predict(ds[test_data,:-1])
    
            scores[i,0] = met.accuracy_score(ds[test_data,-1], test_result)
            scores[i,1] = met.precision_score(ds[test_data,-1], test_result)
            scores[i,2] = met.recall_score(ds[test_data,-1], test_result)
            scores[i,3] = met.f1_score(ds[test_data,-1], test_result)
    
    return {'classifier': classifier.__class__.__name__,'sampler': sampler.__class__.__name__,'accuracy': np.mean(scores[:,0]),'precision': np.mean(scores[:,1]),'recall': np.mean(scores[:,2]),'f1-score': np.mean(scores[:,3])}

if __name__ == '__main__':
    # load dataset from file if exists otherwise build the dataset from database query
    if os.path.isfile('./network_ds.csv'):
        network_ds = pd.read_csv('network_ds.csv', header = None, delimiter = ',', engine = 'python')
        src_net = None
    else:
        network_ds = pd.DataFrame(columns=range(129))
        src_net = pd.read_csv("network.embeddings", header = None, delimiter = ' ', engine = 'python')
        src_net[0] = src_net[0].astype(str)
     
    if os.path.isfile('./task_ds.csv'):
        task_ds = pd.read_csv('task_ds.csv', delimiter = ',', engine = 'python')
    else:
        wm_jobs = WitMartJobs()
        wm_users = WitMartUsers()
        first_date = datetime.strptime('Jan 01, 2011', '%b %d, %Y')
        task_ds = pd.DataFrame(columns=['req_location','req_verified_name','req_followers','req_following','req_tasks','req_completed','req_spending','req_num_rating','req_rating_mean','req_rating_variance','req_work_done','req_first_task','req_last_task','req_avg_bids','req_cancelled','work_location','work_verified_name','work_followers','work_following','work_tasks','work_completed','work_earning','work_num_rating','work_rating_mean','work_rating_variance','work_job_posts','work_first_task','work_last_task','work_awarded','distance','class'])
        ctr = 0
        for job in wm_jobs.get_all():
            data = {}
            requester = wm_users.find_or_insert(job["employer"])
            data['req_location'] = requester['location']
            data['req_verified_name'] = requester['verified_name'] 
            data['req_followers'] = requester['followers']
            data['req_following'] = requester['following']    
            data['req_tasks'] = requester['job_posts']
            # number of completed task
            data['req_completed'] = requester['job_post_completed']
            # total spending
            data['req_spending'] = float(requester['spending'].replace(",","")[1:])
            
            rating = [5 for i in range(requester['job_post_rating_5'])] + [4 for i in range(requester['job_post_rating_4'])] + [3 for i in range(requester['job_post_rating_3'])] + [2 for i in range(requester['job_post_rating_2'])] + [1 for i in range(requester['job_post_rating_1'])]
            # num of rating received
            data['req_num_rating'] = len(rating)
            # avg rating
            data['req_rating_mean'] = np.mean(rating) if len(rating) > 0 else 0
            # var rating
            data['req_rating_variance'] = np.var(rating) if len(rating) > 0 else 0
            # num of work done by the requester
            data['req_work_done'] = requester['work_done']
            # first task posted in days (after 1st january 2011)
            data['req_first_task'] = 0 if requester['job_post_first_bid'] == "" else (datetime.strptime(requester['job_post_first_bid'][:12], '%b %d, %Y') - first_date).days
            # last task posted in days (after 1st january 2011)
            data['req_last_task'] = 0 if requester['job_post_last_bid'] == "" else (datetime.strptime(requester['job_post_last_bid'][:12], '%b %d, %Y') - first_date).days
            # num of avg bids received
            data['req_avg_bids'] = np.mean(requester['job_post_completed_bids']) if len(requester['job_post_completed_bids']) > 0 else 0            
            # num of task cancelled
            data['req_cancelled'] = requester['job_post_cancelled']
            
            req = src_net.loc[np.where(src_net[0] == job["employer"])[0],1:].values
            for bidder in job["bid_list"]:
                    row = copy.deepcopy(data)
                    worker = wm_users.find_or_insert(bidder)
                    if 'name' in worker:
                        row['work_location'] = worker['location']
                        row['work_verified_name'] = worker['verified_name'] # verified real name
                        row['work_followers'] = worker['followers']
                        row['work_following'] = worker['following']
                        # number of work done by worker
                        row['work_tasks'] = worker['work_done']
                        # number of completed work
                        row['work_completed'] = worker['work_done_completed']
                        # total earning
                        row['work_earning'] = float(worker['earning'].replace(",","")[1:])
                        
                        rating = [5 for i in range(worker['work_done_rating_5'])] + [4 for i in range(worker['work_done_rating_4'])] + [3 for i in range(worker['work_done_rating_3'])] + [2 for i in range(worker['work_done_rating_2'])] + [1 for i in range(worker['work_done_rating_1'])]
                        # num of rating received
                        row['work_num_rating'] = len(rating)
                        # avg rating
                        row['work_rating_mean'] = np.mean(rating) if len(rating) > 0 else 0
                        # var rating
                        row['work_rating_variance'] = np.var(rating) if len(rating) > 0 else 0
                        # num of task posted by the worker
                        row['work_job_posts'] = worker['job_posts']
                        # first task done in days (after 1st january 2011)
                        row['work_first_task'] = 0 if worker['work_done_first_bid'] == "" else (datetime.strptime(worker['work_done_first_bid'][:12], '%b %d, %Y') - first_date).days
                        # last task done in days (after 1st january 2011)
                        row['work_last_task'] = 0 if worker['work_done_last_bid'] == "" else (datetime.strptime(worker['work_done_last_bid'][:12], '%b %d, %Y') - first_date).days
                        # num of work awarded
                        row['work_awarded'] = worker['work_done_awarded']
                        
                        # euclidean distance between requester & worker from network
                        work = src_net.loc[np.where(src_net[0] == bidder)[0],1:].values
                        row['distance'] = np.linalg.norm(req - work)
                        # class label
                        row['class'] = 1 if worker['user_id'] in job['winner_list'] else 0
                        task_ds.loc[task_ds.shape[0],:] = row  
                        
                        # for network dataset
                        if src_net is not None:               
                            idx = network_ds.shape[0]
                            network_ds.loc[idx,:127] = np.hstack((req,work))
                            network_ds.loc[idx,128] = row['class']
            ctr += 1
#            print(ctr)
        wm_users.close()    
        wm_jobs.close()
        # save to file
        task_ds.to_csv('task_ds.csv', index = None)
        if src_net is not None:
            network_ds.to_csv('network_ds.csv', index = None, header = None)
    
    
    # create dummy variables for requester & worker locations, and choose only locations with size >1000
    req_locations = pd.get_dummies(task_ds['req_location'],prefix='req')
    filtered_req_loc = np.where(np.sum(req_locations,0) > 1000)[0]
    work_locations = pd.get_dummies(task_ds['work_location'],prefix='work')
    filtered_work_loc = np.where(np.sum(work_locations,0) > 1000)[0]
    
    class_label = task_ds['class']
    # remove non numerical features
    task_ds = task_ds.drop(['req_location','work_location','class'],1)
    # add dummy variables
    task_ds = pd.concat([task_ds,req_locations[req_locations.columns[filtered_req_loc]],work_locations[work_locations.columns[filtered_work_loc]],class_label], axis=1)
    
    # get the indices of each class
    label_1_idx = np.where(network_ds[128] == 1)[0]
    label_0_idx = np.where(network_ds[128] == 0)[0]
    folds = 10
    test_size = 0.05
    model_scores = pd.DataFrame(columns = ['classifier','sampler','accuracy', 'precision', 'recall', 'f1-score'])

    # init the objects for classifier and sampler    
    rf = ens.RandomForestClassifier(n_jobs = -1, n_estimators = 100, max_features = None, criterion = 'entropy', max_depth = 10)   
    gb = ens.GradientBoostingClassifier(n_estimators = 100, max_features = None, loss = 'deviance', max_depth = 10)   
#    nn = net.MLPClassifier(hidden_layer_sizes = (100, 100), activation = 'relu', alpha = 0.001)
#    rus = ius.RandomUnderSampler(replacement=True)
#    ros = ios.RandomOverSampler()
#    nm = ius.NearMiss(n_jobs=-1)
#    ee = ien.EasyEnsemble(n_subsets = 5)
#    bc = ien.BalanceCascade(n_max_subset = 5)

    # build the model for both datasets
    # split the data into train and test set
    # network dataset
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,rf, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,rf, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,rf, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,rf, bc,folds)
#    
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,gb, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,gb, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,gb, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,gb, bc,folds)
#    
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,nn, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,nn, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,nn, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,network_ds.values,nn, bc,folds)
#    
#    # ordinary dataset
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,rf, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,rf, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,rf, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,rf, bc,folds)
#    
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,gb, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,gb, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,gb, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,gb, bc,folds)
#    
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,nn, rus,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,nn, nm,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,nn, ee,folds)
#    model_scores.loc[model_scores.shape[0]] = cv_sampling_imbalanced(label_1_idx,label_0_idx,task_ds.values,nn, bc,folds)
        
    # evaluate the model with different size of samples
    sample_6000_10 = sample_fit_test(network_ds.values,task_ds.values,label_1_idx, label_0_idx,rf,gb,n_sample = 6000, n_iter = 10,folds = 10, test_size = test_size)
    sample_10000_5 = sample_fit_test(network_ds.values,task_ds.values,label_1_idx, label_0_idx,rf,gb,n_sample = 10000, n_iter = 10,folds = 5, test_size = test_size)
    
    # save the result to csv
    desc = np.array([["random forest network 6000 10", "random forest ordinary 6000 10","gradient boosting network 6000 10","gradient boosting ordinary 6000 10","ensemble model both 6000 10","random forest network 10000 5", "random forest ordinary 10000 5","gradient boosting network 10000 5","gradient boosting ordinary 10000 5","ensemble model both 10000 5"]]).T
    cv_results = np.hstack((desc,np.vstack((np.mean(sample_6000_10[0], axis = 1),np.mean(sample_10000_5[0], axis = 1)))))
    test_results = np.hstack((desc,np.vstack((np.mean(sample_6000_10[1], axis = 1),np.mean(sample_10000_5[1], axis = 1)))))
    # first 10 rows are cv scores, next 10 rows are test scores
    results = pd.DataFrame(np.vstack((cv_results,test_results)), columns=["Description", "Accuracy", "Precision", "Recall", "F1-score"])
    results.to_csv("classification_result.csv", index = False)
    print("File classification_result.csv has been generated.")