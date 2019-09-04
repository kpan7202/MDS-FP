# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 10:13:23 2017

@author: Kurniawan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import sklearn.preprocessing as pp
import sklearn.cluster as cls
from datetime import datetime
from collections import Counter
from sklearn.decomposition import PCA
from kmodes import kprototypes as kp
from utilities.witmart import WitMartUsers

dt = pd.read_csv("network.embeddings", header = None, delimiter = ' ', engine = 'python')
num_features = dt.shape[1]
dt.loc[:,num_features] = 0
first_date = datetime.strptime('Jan 01, 2011', '%b %d, %Y')
requester_df = pd.DataFrame(columns= ['id', 'location', 'tasks', 'completed', 'spending', 'num_rating', 'rating_mean', 'rating_variance', 'work_done', 'first_task', 'last_task', 'avg_bids', 'verified_name', 'followers', 'following', 'cancelled' ])
worker_df = pd.DataFrame(columns= ['id', 'location', 'tasks', 'completed', 'earning', 'num_rating', 'rating_mean', 'rating_variance', 'job_posts', 'first_task', 'last_task', 'verified_name', 'followers', 'following', 'awarded' ])
wm_users = WitMartUsers()
for idx, row in dt.iterrows():
    user = wm_users.find_or_insert(str(int(row[0])))
    if 'job_posts' in user and 'work_done' in user:
        data = {}
        data['id'] = user['user_id']
        data['location'] = user['location']
        data['verified_name'] = user['verified_name'] # verified real name
        data['followers'] = user['followers']
        data['following'] = user['following']
        
        if user['job_posts'] > user['work_done']:
            dt.loc[idx, num_features] = 1
            # number of task posted
            data['tasks'] = user['job_posts']
            # number of completed task
            data['completed'] = user['job_post_completed']
            # total spending
            data['spending'] = float(user['spending'].replace(",","")[1:])
            
            rating = [5 for i in range(user['job_post_rating_5'])] + [4 for i in range(user['job_post_rating_4'])] + [3 for i in range(user['job_post_rating_3'])] + [2 for i in range(user['job_post_rating_2'])] + [1 for i in range(user['job_post_rating_1'])]
            # num of rating received
            data['num_rating'] = len(rating)
            # avg rating
            data['rating_mean'] = np.mean(rating) if len(rating) > 0 else 0
            # var rating
            data['rating_variance'] = np.var(rating) if len(rating) > 0 else 0
            # num of work done by the requester
            data['work_done'] = user['work_done']
            # first task posted in days (after 1st january 2011)
            data['first_task'] = 0 if user['job_post_first_bid'] == "" else (datetime.strptime(user['job_post_first_bid'][:12], '%b %d, %Y') - first_date).days
            # last task posted in days (after 1st january 2011)
            data['last_task'] = 0 if user['job_post_last_bid'] == "" else (datetime.strptime(user['job_post_last_bid'][:12], '%b %d, %Y') - first_date).days
            # num of avg bids received
            data['avg_bids'] = np.mean(user['job_post_completed_bids']) if len(user['job_post_completed_bids']) > 0 else 0            
            # num of task cancelled
            data['cancelled'] = user['job_post_cancelled']            
            requester_df.loc[requester_df.shape[0],:] = data
        else:
            # number of work done by worker
            data['tasks'] = user['work_done']
            # number of completed work
            data['completed'] = user['work_done_completed']
            # total earning
            data['earning'] = float(user['earning'].replace(",","")[1:])
            
            rating = [5 for i in range(user['work_done_rating_5'])] + [4 for i in range(user['work_done_rating_4'])] + [3 for i in range(user['work_done_rating_3'])] + [2 for i in range(user['work_done_rating_2'])] + [1 for i in range(user['work_done_rating_1'])]
            # num of rating received
            data['num_rating'] = len(rating)
            # avg rating
            data['rating_mean'] = np.mean(rating) if len(rating) > 0 else 0
            # var rating
            data['rating_variance'] = np.var(rating) if len(rating) > 0 else 0
            # num of task posted by the worker
            data['job_posts'] = user['job_posts']
            # first task done in days (after 1st january 2011)
            data['first_task'] = 0 if user['work_done_first_bid'] == "" else (datetime.strptime(user['work_done_first_bid'][:12], '%b %d, %Y') - first_date).days
            # last task done in days (after 1st january 2011)
            data['last_task'] = 0 if user['work_done_last_bid'] == "" else (datetime.strptime(user['work_done_last_bid'][:12], '%b %d, %Y') - first_date).days
            # num of work awarded
            data['awarded'] = user['work_done_awarded']
            worker_df.loc[worker_df.shape[0],:] = data
    else:
        print("User id not found", row[0])
wm_users.close()

# normalize to 0-1
#scaler = pp.MinMaxScaler()
# standardize to center the mean and scale to unit variance
scaler = pp.StandardScaler()
norm_features = scaler.fit_transform(dt.loc[:, 1:num_features - 1].values)
requester_norm_features = scaler.fit_transform(requester_df.iloc[:, 2:16].values)
worker_norm_features = scaler.fit_transform(worker_df.iloc[:, 2:15].values)

# use PCA for visualisation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(norm_features)
requester_pca_result = pca.fit_transform(requester_norm_features)
worker_pca_result = pca.fit_transform(worker_norm_features)

# plot true labels
pl.figure()
pl.scatter(pca_result[np.where(dt.loc[:,num_features] == 1)[0] ,0], pca_result[np.where(dt.loc[:,num_features] == 1)[0],1], label = "requesters " + str(len(np.where(dt.loc[:,num_features] == 1)[0])))
pl.scatter(pca_result[np.where(dt.loc[:,num_features] == 0)[0] ,0], pca_result[np.where(dt.loc[:,num_features] == 0)[0],1], label = "workers " + str(len(np.where(dt.loc[:,num_features] == 0)[0])))
pl.title("True Labels")
pl.legend()
pl.show()

# cluster deepwalk features
for i in range(2,5):
#    network = cls.KMeans(n_clusters= i, n_jobs= -1)
    network = cls.Birch(n_clusters=i, threshold=10, branching_factor=100)
    labels = network.fit_predict(norm_features)
    dt[num_features - 1 + i] = labels
    
    pl.figure()
    for j in range(i):
        pl.scatter(pca_result[np.where(dt.loc[:, num_features - 1 + i] == j)[0] ,0], pca_result[np.where(dt.loc[:, num_features - 1 + i] == j)[0],1], label = str(j) + ": " + str(len(np.where(dt.loc[:, num_features - 1 + i] == j)[0])))
    pl.title("Clustering result k=" + str(i))
    pl.legend()
    pl.show()

# birch clustering    
birch = cls.Birch(threshold=4,branching_factor=100, n_clusters=None) #cls.SpectralClustering(n_clusters=2, n_jobs=-1) #cls.AgglomerativeClustering(affinity="cosine", linkage="average") #cls.DBSCAN(eps = 2, min_samples=20, n_jobs=-1)
requester_df['birch_label'] = birch.fit_predict(requester_norm_features)
pl.figure()
for j in Counter(requester_df['birch_label']):
    pl.scatter(requester_pca_result[np.where(requester_df['birch_label'] == j)[0] ,0], requester_pca_result[np.where(requester_df['birch_label'] == j)[0], 1], label = str(j) + ": " + str(len(np.where(requester_df['birch_label'] == j)[0])))
pl.title("Requesters Birch Clustering")
pl.legend()
pl.show()

agglo = cls.AgglomerativeClustering(n_clusters=2,affinity="cosine", linkage="average")#cls.Birch(threshold=5,branching_factor=1000, n_clusters=3) #cls.SpectralClustering(n_clusters=2, n_jobs=-1)  #cls.DBSCAN(eps = 2, min_samples=100, n_jobs=-1) 
worker_df['agglo_label'] = agglo.fit_predict(worker_norm_features)
pl.figure()
for j in Counter(worker_df['agglo_label']):
    pl.scatter(worker_pca_result[np.where(worker_df['agglo_label'] == j)[0] ,0], worker_pca_result[np.where(worker_df['agglo_label'] == j)[0], 1], label = str(j) + ": " + str(len(np.where(worker_df['agglo_label'] == j)[0])))
pl.title("Workers Agglomerative Clustering")
pl.legend()
pl.show()
    
# kprototypes clustering
requester_norm_df = pd.DataFrame(np.hstack((requester_df.iloc[:,1:2].values, requester_norm_features)))
worker_norm_df = pd.DataFrame(np.hstack((worker_df.iloc[:,1:2].values, worker_norm_features)))
for i in range(2,6): 
    kproto = kp.KPrototypes(n_clusters=i)
    # cluster requesters data   
    requester_label = kproto.fit_predict(requester_norm_df.iloc[:,1:].values, categorical=[0])
    requester_df['kmeans_'+ str(i)] = requester_label
    
    pl.figure()
    for j in range(i):
        pl.scatter(requester_pca_result[np.where(requester_df["kmeans_" + str(i)] == j)[0] ,0], requester_pca_result[np.where(requester_df["kmeans_" + str(i)] == j)[0], 1], label = str(j) + ": " + str(len(np.where(requester_df["kmeans_" + str(i)] == j)[0])))
    pl.title("Requesters Clustering result k=" + str(i))
    pl.legend()
    pl.show()

    # cluster workers data 
    worker_label = kproto.fit_predict(worker_norm_df.iloc[:,1:].values, categorical=[0])
    worker_df['kmeans_'+ str(i)] = worker_label
        
    pl.figure()
    pl.scatter(worker_pca_result[:,0], worker_pca_result[:,1], c=worker_label)
    for j in range(i):
        pl.scatter(worker_pca_result[np.where(worker_df["kmeans_" + str(i)] == j)[0] ,0], worker_pca_result[np.where(worker_df["kmeans_" + str(i)] == j)[0], 1], label = str(j) + ": " + str(len(np.where(worker_df["kmeans_" + str(i)] == j)[0])))
    pl.title("Workers Clustering result k=" + str(i))
    pl.legend()
    pl.show()
    
   