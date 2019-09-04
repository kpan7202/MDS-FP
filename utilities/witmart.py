# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:20:50 2017

@author: Kurniawan
"""

import sys
import pymongo

# class to handle operation on collection "jobs"
class WitMartJobs():
    connection = None
    def __init__(self):
        try:
            self.connection = pymongo.MongoClient("mongodb://localhost:27017/")
        except:
            print("Error:", sys.exc_info()[0])
            
    def insert_job(self, data):
        try:
            jobs_col = self.connection.fp.jobs
            return jobs_col.insert_one(data).inserted_id
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def update_job(self, data):
        try:
            jobs_col = self.connection.fp.jobs
            jobs_col.save(data)
        except:
            print("Error:", sys.exc_info()[0])
    
    def find_job_by_id(self, job_id):
        try:
            jobs_col = self.connection.fp.jobs
            return jobs_col.find_one({'job_id': job_id})
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def get_all(self):
        try:
            jobs_col = self.connection.fp.jobs
            return jobs_col.find().sort([('job_id', pymongo.ASCENDING)])
        except:
            print("Error:", sys.exc_info()[0])
        return None
        
    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
        except:
            print("Error:", sys.exc_info()[0])

# class to handle operation on collection "users"
class WitMartUsers():
    connection = None
    def __init__(self):
        try:
            self.connection = pymongo.MongoClient("mongodb://localhost:27017/")
        except:
            print("Error:", sys.exc_info()[0])
            
    def find_or_insert(self, user_id):
        try:
            u = {'user_id': user_id}
            users_col = self.connection.fp.users
            user = users_col.find_one(u)
            if user is None:
                _id = users_col.insert_one(u).inserted_id
                user = u
                user['_id'] = _id
            return user
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def update_user(self, user):
        try:
            users_col = self.connection.fp.users
            users_col.save(user)
        except:
            print("Error:", sys.exc_info()[0])
            
    def get_all(self):
        try:
            users_col = self.connection.fp.users
            return users_col.find().sort([('user_id', pymongo.ASCENDING)])
        except:
            print("Error:", sys.exc_info()[0])
        return None
            
    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
        except:
            print("Error:", sys.exc_info()[0])

# class to handle operation on collection "connection"            
class WitMartConnection():
    connection = None
    def __init__(self):
        try:
            self.connection = pymongo.MongoClient("mongodb://localhost:27017/")
        except:
            print("Error:", sys.exc_info()[0])
            
    def insert_connection(self, data, t = None):
        try:
            conn_col = self.get_collection(t)
            return conn_col.insert_one(data).inserted_id
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def update_connection(self, data, t = None):
        try:
            conn_col = self.get_collection(t)
            conn_col.save(data)
        except:
            print("Error:", sys.exc_info()[0])
    
    def find_connection(self, poster_id, worker_id, t = None):
        try:
            conn_col = self.get_collection(t)
            return conn_col.find_one({'poster_id': poster_id, 'worker_id': worker_id})
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def get_connection(self, t = None, params = {}):
        try:
            conn_col = self.get_collection(t)
            return conn_col.find(params).sort([('worker_id', pymongo.ASCENDING)])
        except:
            print("Error:", sys.exc_info()[0])
        return None
    
    def get_unique_user_ids(self, t = None):
        user_ids = set()
        try:
            conn_col = self.get_collection(t)
            for uid in conn_col.distinct("poster_id"):
                user_ids.add(uid)
            for uid in conn_col.distinct("worker_id"):
                user_ids.add(uid)
        except:
            print("Error:", sys.exc_info()[0])
        return user_ids
    
    def get_collection(self, t = None):
        if t is None:
            return self.connection.fp.connection
        elif t == "LD":
            return self.connection.fp.logo_design
        elif t == "T":
            return self.connection.fp.translation
        elif t == "WD":
            return self.connection.fp.web_design
        elif t == "AD":
            return self.connection.fp.app_development
        elif t == "SM":
            return self.connection.fp.sales_marketing
        elif t == "OS":
            return self.connection.fp.other_services
            
    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
        except:
            print("Error:", sys.exc_info()[0])
           
            