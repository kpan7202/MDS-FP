# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:26:59 2017

@author: Kurniawan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
from collections import Counter, OrderedDict
from scipy.stats import norm, gaussian_kde

network_data = pd.read_csv("network.csv")
requester_df = pd.read_csv("requester.csv")
worker_df = pd.read_csv("worker.csv")

# requesters from network
nr1 = network_data.loc[np.where(network_data.loc[:,"c3"] == 0)[0], "0"]
nr2 = network_data.loc[np.where(network_data.loc[:,"c3"] == 2)[0], "0"]
# requesters by birch clustering
br1 = requester_df.loc[np.where(requester_df.loc[:,"birch_label"] == 1)[0], "id"]
br2 = requester_df.loc[np.where(requester_df.loc[:,"birch_label"] == 3)[0], "id"]
# requesters by kmeans 3 clustering
kr1 = requester_df.loc[np.where(requester_df.loc[:,"kmeans_3"] == 0)[0], "id"]
kr2 = requester_df.loc[np.where(requester_df.loc[:,"kmeans_3"] == 2)[0], "id"]

#network analysis between true label and kmeans 2
network_label_diff = np.zeros((2,2), int)
for i in range(2): #row - true label
    tids = np.where(network_data.loc[:,"true label"] == i)[0]
    for j in range(2): #col - kmeans 2
        kids = np.where(network_data.loc[:,"c2"] == j)[0]
        network_label_diff[i,j] = len(np.intersect1d(tids, kids, True))

# find the proportion of requester ids from network data and requester data 
# compare with k means-3 and birch clustering
# row indicates network label 1 & 2
# column indicates kmeans-3 label 1 & 2 and birch label 1 & 2
diff = np.zeros((2,4), int)
diff[0,0] = len(np.intersect1d(nr1, kr1, True))
diff[0,1] = len(np.intersect1d(nr1, kr2, True))
diff[1,0] = len(np.intersect1d(nr2, kr1, True))
diff[1,1] = len(np.intersect1d(nr2, kr2, True))
diff[0,2] = len(np.intersect1d(nr1, br1, True))
diff[0,3] = len(np.intersect1d(nr1, br2, True))
diff[1,2] = len(np.intersect1d(nr2, br1, True))
diff[1,3] = len(np.intersect1d(nr2, br2, True))
diff = pd.DataFrame(diff, index = ["network 0", "network 2"], columns=["k3 0", "k3 2", "birch 1", "birch 3"])


def plotBar(data, title = "", size = (6,6)):
    pl.figure(figsize=size)
    pl.bar(range(len(data.keys())),data.values())
    pl.xticks(range(len(data.keys())), data.keys())
    pl.title(title)
    pl.show()

def plotDensity(data, n_bins, title = "", xlabel = "", ylabel = ""):
    n, bins, patches = pl.hist(data, n_bins, normed=1, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, np.mean(data), np.std(data))
    pl.plot(bins, y, 'r--')
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(title)
    pl.show()

# analyze the distribution of some attributes
# requester number of tasks
temp = Counter(requester_df["tasks"])
req_task = OrderedDict()
for key in sorted(temp):
    req_task[key] = temp[key]
plotBar(req_task, "Number of Tasks per Requester")

# requester number of completed tasks
temp = Counter(requester_df["completed"])
req_completed = OrderedDict()
for key in sorted(temp):
    req_completed[key] = temp[key]
plotBar(req_completed,"Number of Completed Tasks per Requester")

# requester number of task vs completed
pl.scatter(requester_df["tasks"], requester_df["completed"])
pl.title("Requester Number of Task vs Completed")
pl.xlabel("Number of Task")
pl.ylabel("Number of Completed Task")
pl.show()

# requester sum of rating
sum_rating = requester_df["num_rating"].values * requester_df["rating_mean"].values
sorted_val = sorted(sum_rating.astype(int))
temp = Counter(sorted_val)
sum_rating_ctr = OrderedDict()
for key in sorted(temp):
    sum_rating_ctr[key] = temp[key]
plotBar(sum_rating_ctr,"Sum of Rating", (9,6))
plotDensity(sorted_val, np.arange(np.min(sorted_val), np.max(sorted_val) + 1000, 500), 'Histogram & PDF', 'Sum of Rating', 'Probability')

# requester mean of rating
plotDensity(requester_df["rating_mean"], 5, 'Histogram & PDF', 'Average of Rating', 'Probability')

# requester money spent
# ignore 0 value
spending = requester_df.loc[np.where(requester_df["spending"] > 0)[0],"spending"]
plotDensity(spending, np.arange(np.min(spending), np.max(spending) + 10000, 5000), 'Histogram & PDF', 'Money Spent', 'Probability')

# requester first task posted
valid_year = requester_df.loc[np.where(requester_df["first_task"] != -1)[0],"first_task"]
temp = Counter(valid_year)
year_first_task = OrderedDict()
for key in sorted(temp):
    year_first_task[key] = temp[key]
plotBar(year_first_task, "First Task Posted")
plotDensity(valid_year, np.arange(2011, 2019, 1), 'Histogram & PDF', 'Year', 'Probability')

# requester last task posted
valid_year = requester_df.loc[np.where(requester_df["last_task"] != -1)[0],"last_task"]
temp = Counter(valid_year)
year_last_task = OrderedDict()
for key in sorted(temp):
    year_last_task[key] = temp[key]
plotBar(year_last_task, "Last Task Posted")
plotDensity(valid_year, np.arange(2011, 2019, 1), 'Histogram & PDF', 'Year', 'Probability')


# requester years active
year_active = requester_df["last_task"] - requester_df["first_task"]
valid_year = year_active[(year_active > -1) & (year_active < 10)]
temp = Counter(valid_year)
year_active = OrderedDict()
for key in sorted(temp):
    year_active[key] = temp[key]
plotBar(year_active, "Years Active")
plotDensity(valid_year, np.arange(0, 6, 1), 'Histogram & PDF', 'Year', 'Probability')

# requester location
req_location = Counter(requester_df["location"])
plotBar(req_location, "Requester Location", (10,6))

#worker
# worker number of tasks
temp = Counter(worker_df["tasks"])
work_task = OrderedDict()
for key in sorted(temp):
    work_task[key] = temp[key]
plotBar(work_task, "Number of Tasks per Worker", (10, 6))

# worker number of completed tasks
temp = Counter(worker_df["completed"])
work_completed = OrderedDict()
for key in sorted(temp):
    work_completed[key] = temp[key]
plotBar(work_completed,"Number of Tasks Done by Worker", (10, 6))

# worker number of task vs completed
pl.scatter(worker_df["tasks"], worker_df["completed"])
pl.title("Worker Number of Work vs Completed")
pl.xlabel("Number of Work")
pl.ylabel("Number of Completed Work")
pl.show()

# worker sum of rating
sum_rating = worker_df["num_rating"].values * worker_df["rating_mean"].values
sorted_val = sorted(sum_rating.astype(int))
temp = Counter(sorted_val)
sum_rating_ctr = OrderedDict()
for key in sorted(temp):
    sum_rating_ctr[key] = temp[key]
plotBar(sum_rating_ctr,"Sum of Rating", (9,6))
plotDensity(sorted_val, np.arange(np.min(sorted_val), np.max(sorted_val) + 1000, 500), 'Histogram & PDF', 'Sum of Rating', 'Probability')

# worker mean of rating
plotDensity(worker_df["rating_mean"], 5, 'Histogram & PDF', 'Average of Rating', 'Probability')

# worker money earned
# ignore 0 value
earning = worker_df.loc[np.where(worker_df["earning"] > 0)[0],"earning"]
plotDensity(earning, np.arange(np.min(earning), np.max(earning) + 10000, 10000), 'Histogram & PDF', 'Money Spent', 'Probability')

# worker first task done
valid_year = worker_df.loc[np.where(worker_df["first_task"] != -1)[0],"first_task"]
temp = Counter(valid_year)
year_first_task = OrderedDict()
for key in sorted(temp):
    year_first_task[key] = temp[key]
plotBar(year_first_task, "First Task Done")
plotDensity(valid_year, np.arange(2011, 2019, 1), 'Histogram & PDF', 'Year', 'Probability')

# worker last task done
valid_year = worker_df.loc[np.where(worker_df["last_task"] != -1)[0],"last_task"]
temp = Counter(valid_year)
year_last_task = OrderedDict()
for key in sorted(temp):
    year_last_task[key] = temp[key]
plotBar(year_last_task, "Last Task Done")
plotDensity(valid_year, np.arange(2011, 2019, 1), 'Histogram & PDF', 'Year', 'Probability')

# worker years active
year_active = worker_df["last_task"] - worker_df["first_task"]
valid_year = year_active[(year_active > -1) & (year_active < 10)]
temp = Counter(valid_year)
year_active = OrderedDict()
for key in sorted(temp):
    year_active[key] = temp[key]
plotBar(year_active, "Years Active")
plotDensity(valid_year, 6, 'Histogram & PDF', 'Year', 'Probability')

# worker location
work_location = Counter(worker_df["location"])
plotBar(work_location, "Worker Location", (10,6))