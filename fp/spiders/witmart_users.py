# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:20:50 2017

@author: Kurniawan

A class to crawl Witmart.com user info page
"""

import scrapy
from utilities.util import strip_tags_spaces
from utilities.witmart import WitMartJobs, WitMartUsers, WitMartConnection


class WitmartUsersSpider(scrapy.Spider):
    name = 'witmart-users'
    url = 'http://www.witmart.com'
    
    # default method that serves as a starting point for crawler
    def start_requests(self):
		# try to get all jobs stored in the db
        wm_users = WitMartUsers()
#        users = wm_users.get_all()
        wm_conn = WitMartConnection()
        for user_id in wm_conn.get_unique_user_ids("LD"):
            user = wm_users.find_or_insert(user_id)
            if 'name' not in user: # new record
			# iterate for each user and grab user id
#            for user in users: 
#                user_id = user['user_id']
                param = {'user_id': user_id}
                yield scrapy.Request(url = self.url + '/u/' + user_id + '/jobposts', callback=self.parse_job_post, meta=param)
                yield scrapy.Request(url = self.url + '/u/' + user_id + '/workhistory', callback=self.parse_work_history, meta=param)

        wm_users.close()       
        wm_conn.close()
        
    
    # method to parse user info page, as a job poster
    def parse_job_post(self, response):
        user_id = response.meta.get('user_id')
        wm_users = WitMartUsers()
        # if user data is found in the db then query the data otherwise insert new record and return the newly created data.
        user = wm_users.find_or_insert(user_id)
        
        user['name'] = response.css('div.user-show_r h1::text').extract_first()
        user['location'] = response.css('div.user-show_r span span::text').extract_first()
        user['membership'] = response.css('div.user-show_r a.goldshow span::text').extract_first()
        user['verified_name'] = 1 if response.css('i.Verification4_1').extract_first() is not None or response.css('i.Verification5_1').extract_first() is not None else 0
        user['verified_email'] = 1 if response.css('i.Verification2_1').extract_first() is not None else 0
        user['verified_phone'] = 1 if response.css('i.Verification3_1').extract_first() is not None else 0
        
        followers = 0
        following = 0
        for s in response.css('div.follow_n a'):
            href = s.css('a::attr(href)').extract_first()
            val = int(s.css('a::text').extract_first())
            if href.find('followers') >= 0:
                followers = val
            elif href.find('following') >= 0:
                following = val
        user['followers'] = followers
        user['following'] = following
        
        user['professions'] = []
        occ = response.css('div.user-show_r dd ul').extract_first()
        if occ is not None:
            occ_list = strip_tags_spaces(occ).strip()
            if occ_list != "":
                user['professions'] = occ_list.replace('Professions: ', '').split(' , ')
        
        # parse and collect rating data if exist
        review = response.css('div.job-list h3').extract_first()
        if review is not None and "Rating" in review:
            job_posts = response.css('div.job-list h3::text').extract_first()
            user['job_posts'] = 0 if job_posts is None else int(job_posts[24 : -1])
            for rating in response.css('div.job-list_re ul')[0].css('li'):
                temp = rating.css('a::text').extract_first()[0]
                user['job_post_rating_' + temp] = int(rating.css('span::text').extract_first())
            
            user['job_post_completed'] = int(response.css('div.job-list_re ul')[1].css('li')[1].css('a::text').extract_first().split(' ')[0])
            user['job_post_cancelled'] = int(response.css('div.job-list_re ul')[1].css('li')[2].css('a::text').extract_first().split(' ')[0])
            user['spending'] = '$0' if user['job_posts'] == 0 else response.css('div.job-list_re p::text').extract_first()
        else: # set to default values
            user['job_posts'] = 0
            user['job_post_rating_5'] = 0
            user['job_post_rating_4'] = 0
            user['job_post_rating_3'] = 0
            user['job_post_rating_2'] = 0
            user['job_post_rating_1'] = 0
            user['job_post_completed'] = 0
            user['job_post_cancelled'] = 0
            user['spending'] = '$0'
            
        # parse list of jobs posted, to collect first & last job posts and bids made
        user['job_post_first_bid'] = ""
        user['job_post_last_bid'] = ""
        user['job_post_completed_bids'] = []
        for url in response.css('ul.jinjobs li'):
            link = url.css('a.ajobs::attr(href)').extract_first()
            if link is not None and link.find('user/sign/in') == -1 and link.find('/logo-design/') >= 0:
                temp = url.css('p')[0].extract()
                if temp.find("Status: Completed") >= 0:
                    attrs = strip_tags_spaces(temp).split('|')
                    user['job_post_completed_bids'].append(int(attrs[3].replace("Bids", "").strip()))
                    if user['job_post_last_bid'] == "":
                        user['job_post_last_bid'] = attrs[0].strip()                   
                    user['job_post_first_bid'] = attrs[0].strip()  
                
        # update user data
        wm_users.update_user(user)
        wm_users.close()
        
        next_page = response.css('div.oturning span.st4 a.reviews::attr(href)').extract_first()
        if next_page is not None:
            param = {'user_id': user_id, 'type': 'jp'}
            yield response.follow(next_page, callback=self.parse_next, meta=param)
        
    # method to parse user info page, as a worker    
    def parse_work_history(self, response):
        user_id = response.meta.get('user_id')
        wm_users = WitMartUsers()
        # if user data is found in the db then query the data otherwise insert new record and return the newly created data.
        user = wm_users.find_or_insert(user_id)
		
        # parse and collect rating data if exist
        review = response.css('div.job-list h3').extract_first()
        if review is not None and "Rating" in review:
            work_done = response.css('div.job-list h3::text').extract_first()
            user['work_done'] = 0 if work_done is None else int(work_done[26 : -1])
            for rating in response.css('div.job-list_re1 ul')[0].css('li'):
                temp = rating.css('a::text').extract_first()[0]
                user['work_done_rating_' + temp] = int(rating.css('span::text').extract_first())
            
            user['work_done_awarded'] = int(response.css('div.job-list_re1 ul')[1].css('li')[0].css('a::text').extract_first().split(' ')[0])
            user['work_done_completed'] = int(response.css('div.job-list_re1 ul')[1].css('li')[1].css('a::text').extract_first().split(' ')[0])
            user['earning'] = '$0' if user['work_done'] == 0 else response.css('div.job-list_re1 p::text').extract_first()
        else: # set to default values
            user['work_done'] = 0
            user['work_done_rating_5'] = 0
            user['work_done_rating_4'] = 0
            user['work_done_rating_3'] = 0
            user['work_done_rating_2'] = 0
            user['work_done_rating_1'] = 0
            user['work_done_awarded'] = 0
            user['work_done_completed'] = 0
            user['earning'] = '$0'
        
        # parse first & last work history bids made
        user['work_done_first_bid'] = ""
        user['work_done_last_bid'] = ""
        for url in response.css('ul.jinjobs li'):
            link = url.css('a.ajobs::attr(href)').extract_first()
            if link is not None and link.find('user/sign/in') == -1 and link.find('/logo-design/') >= 0:
                temp = url.css('p')[0].extract()
                if temp.find("Status: Completed") >= 0:
                    attrs = strip_tags_spaces(temp).split('|')
                    if user['work_done_first_bid'] == "":
                        user['work_done_first_bid'] = attrs[0].strip()                   
                    user['work_done_last_bid'] = attrs[0].strip()  
                    
        # update user data
        wm_users.update_user(user)
        wm_users.close()
        
        next_page = response.css('div.oturning span.st4 a.reviews::attr(href)').extract_first()
        if next_page is not None:
            param = {'user_id': user_id, 'type': 'wh'}
            yield response.follow(next_page, callback=self.parse_next, meta=param)

    
    # method to parse task list and to follow each link
    def parse_next(self, response):
        user_id = response.meta.get('user_id')
        user_type = response.meta.get('type')
        
        wm_users = WitMartUsers()
        # if user data is found in the db then query the data otherwise insert new record and return the newly created data.
        user = wm_users.find_or_insert(user_id)
        if 'name' in user:
            # for each url to task details page, parse info regarding first & last bids made for both job posts and work history
            for url in response.css('ul.jinjobs li'):
                link = url.css('a.ajobs::attr(href)').extract_first()
                if link is not None and link.find('user/sign/in') == -1 and link.find('/logo-design/') >= 0:
                    temp = url.css('p')[0].extract()
                    if temp.find("Status: Completed") >= 0:
                        attrs = strip_tags_spaces(temp).split('|')
                        if user_type == 'jp':
                            user['job_post_completed_bids'].append(int(attrs[3].replace("Bids", "").strip()))
                            if user['job_post_last_bid'] == "":
                                user['job_post_last_bid'] = attrs[0].strip()                   
                            user['job_post_first_bid'] = attrs[0].strip()
                        else:
                            if user['work_done_first_bid'] == "":
                                user['work_done_first_bid'] = attrs[0].strip()                   
                            user['work_done_last_bid'] = attrs[0].strip()  
                    
            # update user data
            wm_users.update_user(user)
            
        wm_users.close()
            
        # if there is next page, then go to next page
        next_page = response.css('div.oturning span.st4 a.reviews::attr(href)').extract_first()
        if next_page is not None:
            param = {'user_id': user_id, 'type': user_type}
            yield response.follow(next_page, callback=self.parse_next, meta=param)
        
