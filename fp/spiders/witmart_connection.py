# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:54:50 2017

@author: Kurniawan

Class to crawl witmart task detail and user info pages, to build user network
"""


import scrapy
from utilities.util import strip_tags_spaces
from utilities.witmart import WitMartJobs, WitMartUsers, WitMartConnection


class WitmartConnectionSpider(scrapy.Spider):
    name = 'witmart-connection'
    url = 'http://www.witmart.com'
    init = 10
    
    # default method as a starting point for crawler
    def start_requests(self):
        # level indicates max depth of the cycle for the crawler to go, set to -1 for no limit
        level = 2
        param = {'level' : level}               
        yield scrapy.Request(url = self.url + '/logo-design/jobs?s=3', callback=self.parse, meta=param)
        
    # default method to handle parsing the urls stored in start_urls
    def parse(self, response):
        param = {'level' : response.meta.get('level')}
        for job in response.css('ul#joblist li'):   
            # for each url if it's not private job (must sign in to see) then follow the link
            link = job.css('a.fromtitle::attr(href)').extract_first()
            if link is not None and link.find('user/sign/in') == -1:
#                self.init -= 1
                yield response.follow(link, callback=self.parse_job, meta = param)
            if  self.init <= 0:
                break

        # parse the next page link if it exists
        next_page = response.css('span.st4 a::attr(href)').extract_first()
        if next_page is not None and self.init > 0:
            yield response.follow(next_page, callback=self.parse, meta = param)
        
    # method to parse task detail page
    def parse_job(self, response):
        level = response.meta.get('level')
        job_id = response.css('input#jobid::attr(value)').extract_first()
        wm_jobs = WitMartJobs()
        if job_id is not None: 
#            job_data = wm_jobs.find_job_by_id(job_id)
#            if  job_data is None:
            data = {}
            data['job_id'] = job_id
            data['title'] = response.css('div.gj_title h2::text').extract_first().strip()
            
            temp = response.css('div.gj_title p.dq_nav a')       
            data['employer'] = temp[0].css('a::attr(href)').extract_first()[3:]
            data['category'] = temp[1].css('a::text').extract_first().strip()
            data['type'] = temp[2].css('a::text').extract_first().strip()
            
            temp = response.css('table.t_details tr') 
            data['status'] = strip_tags_spaces(temp[0].css('td')[0].css('td').extract_first())[12:] # remove 'Job Status: '
            
            if data['status'] == 'Completed' and data['category'] == 'Graphic & Logo  Design':
                if temp[1].css('td')[0].css('b.g-f14::text').extract_first().find('eward') >= 0:
                    data['reward'] = strip_tags_spaces(temp[1].css('td')[0].css('td strong::text').extract_first())
                else:
                    reward = 'Negotiable'
                    if len(temp) >= 3:
                        reward = strip_tags_spaces(temp[2].css('td').extract_first())
                        idx = reward.find(':')
                        if idx >= 0:
                            reward = reward[idx + 2:].strip()
                    data['reward'] = reward
                
                data['bid_start'] = strip_tags_spaces(temp[0].css('td')[1].css('td').extract_first())[17:] #remove 'Bidding Started: '
                data['bid_end'] = strip_tags_spaces(temp[1].css('td')[1].css('td').extract_first())[15:] #remove 'Bidding Ended: '
                
                if response.css('div#j-langdes').extract_first() is not None:
                    temp = strip_tags_spaces(response.css('div#j-langdes').extract_first()).strip()
                    if temp == "" and response.css('div#j-hidefortrans').extract_first() is not None:
                        attrs = response.css('div#j-hidefortrans h5::text').extract()
                        values = response.css('div#j-hidefortrans div.JOBDESC').extract()
                        for kvp in zip(attrs,values):
                            temp += kvp[0] + ": " + strip_tags_spaces(kvp[1]) + "; "           
                    data['description'] = temp
                    
                data['bid_list'] = []
                data['winner_list'] = []
                for bidder in response.css('div#all_bids dl.list'):
                    user_id = bidder.css('dd.col1::attr(value)').extract_first()
                    data['bid_list'].append(user_id)
                    winner = False
                    if bidder.css('dd.zb').extract_first() is not None:
                        data['winner_list'].append(user_id)
                        winner = True
                    # call create_connection() method for each bidder
                    for req in self.create_connection(level, job_id, data['employer'], user_id, winner):
                        yield req
                    
                data['required_skills'] = []
                for skill in response.css('div.t_des div.mt20 a'):
                    data['required_skills'].append(skill.css('a::text').extract_first())
                
                # temporary to update records
                job_data = wm_jobs.find_job_by_id(job_id)
                if  job_data is None:                
                    wm_jobs.insert_job(data)
                else:
                    wm_jobs.update_job(data)
            
                next_bid_page = response.css('i.next a::attr(href)').extract_first()
                if next_bid_page is not None:
                    yield response.follow(next_bid_page, callback=self.parse_bid, meta={'level': level})
#            else:
#                for bidder in job_data['bid_list']:
#                   #add logo design relationship
#                   winner = False
#                   if bidder in job_data['winner_list']:
#                       winner = True
#                   for req in self.create_connection(level, job_data['job_id'], job_data['employer'], bidder, winner):
#                       yield req
        wm_jobs.close()

    # method to parse bidder list to get user ids if there are more than 10 bidders for a task    
    def parse_bid(self, response):
        level = response.meta.get('level')
        job_id = response.css('input#jobid::attr(value)').extract_first()
        wm_jobs = WitMartJobs()
        data = wm_jobs.find_job_by_id(job_id)
        if data is not None:
            for bidder in response.css('div#all_bids dl.list'):
                user_id = bidder.css('dd.col1::attr(value)').extract_first()
                if user_id not in data['bid_list']:
                    data['bid_list'].append(user_id)
                winner = False
                if bidder.css('dd.zb').extract_first() is not None and user_id not in data['winner_list']:
                    data['winner_list'].append(user_id)
                    winner = True
                for req in self.create_connection(level, job_id, data['employer'], user_id, winner):
                    yield req
            wm_jobs.update_job(data)
         
            next_bid_page = response.css('i.next a::attr(href)').extract_first()
            if next_bid_page is not None:
                yield response.follow(next_bid_page, callback=self.parse_bid, meta={'level': level})
                
        wm_jobs.close()
        
    # method to insert / update connection between employer and worker    
    def create_connection(self, level, job_id, poster_id, bidder_id, winner = False):    
        wm_conn = WitMartConnection()
        conn = wm_conn.find_connection(poster_id, bidder_id, "LD")
        if conn is None:
            conn = {}
            conn['poster_id'] = poster_id
            conn['worker_id'] = bidder_id
            conn['job_ids'] = [job_id]
            conn['winner_job_ids'] = [job_id] if winner == True else []
            wm_conn.insert_connection(conn, "LD")
        elif job_id not in conn['job_ids']:
            conn['job_ids'].append(job_id)
            if winner == True:
                conn['winner_job_ids'].append(job_id)
            wm_conn.update_connection(conn, "LD")
        wm_conn.close()
        
        # if level is still positive number or -1 the proceed to follow next cycle
        if level == -1 or level > 0:    
            newLevel = -1 if level == -1 else level - 1
            wm_users = WitMartUsers()
            # crawl employer info
            user = wm_users.find_or_insert(poster_id)
            if 'name' not in user: # new record
#            if True: #temporary to update records
                param = {'user_id': poster_id, 'level' : newLevel}
                yield scrapy.Request(url = self.url + '/u/' + poster_id + '/jobposts', callback=self.parse_poster, meta=param)
                yield scrapy.Request(url = self.url + '/u/' + poster_id + '/workhistory', callback=self.parse_worker, meta=param)
            # crawl worker info
            user = wm_users.find_or_insert(bidder_id)
            if 'name' not in user: # new record
#            if True: #temporary to update records
                param = {'user_id': bidder_id, 'level' : newLevel}
                yield scrapy.Request(url = self.url + '/u/' + bidder_id + '/jobposts', callback=self.parse_poster, meta=param)
                yield scrapy.Request(url = self.url + '/u/' + bidder_id + '/workhistory', callback=self.parse_worker, meta=param)
        
            wm_users.close()
            
    # method to parse user info page for job poster role        
    def parse_poster(self, response):
        user_id = response.meta.get('user_id')
        wm_users = WitMartUsers()
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
        else:
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
            
        wm_users.update_user(user)
        wm_users.close()
        
        # set parameter for initial call
        params = {'user_id': user_id, 'user_type': 'jp'}
        for req in self.parse_task(response, params):
            yield req
        
    # method to parse user info page for worker role     
    def parse_worker(self, response):
        user_id = response.meta.get('user_id')
        wm_users = WitMartUsers()
        user = wm_users.find_or_insert(user_id)
        
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
        else:
            user['work_done'] = 0
            user['work_done_rating_5'] = 0
            user['work_done_rating_4'] = 0
            user['work_done_rating_3'] = 0
            user['work_done_rating_2'] = 0
            user['work_done_rating_1'] = 0
            user['work_done_awarded'] = 0
            user['work_done_completed'] = 0
            user['earning'] = '$0'
            
        user['work_done_first_bid'] = ""
        user['work_done_last_bid'] = ""
        
        wm_users.update_user(user)
        wm_users.close()
        
        # set parameter for initial call
        params = {'user_id': user_id, 'user_type': 'wh'}
        for req in self.parse_task(response, params):
            yield req
        
    # method to parse task list and to follow each link
    def parse_task(self, response, params = None):
        level = response.meta.get('level')
        user_id = params['user_id'] if params is not None else response.meta.get('user_id')
        user_type = params['user_type'] if params is not None else response.meta.get('user_type')
            
        wm_users = WitMartUsers()
        # if user data is found in the db then query the data otherwise insert new record and return the newly created data.
        user = wm_users.find_or_insert(user_id)
        
        links = []
        # for each url to task details page, parse info regarding first & last bids made for both job posts and work history
        for url in response.css('ul.jinjobs li'):
            link = url.css('a.ajobs::attr(href)').extract_first()
            if link is not None and link.find('user/sign/in') == -1 and link.find('/logo-design/') >= 0:
                links.append(link)
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
        
        for link in links:
            yield response.follow(link, callback=self.parse_job, meta={'level': level})
                
        # if there is next page, then go to next page
        next_page = response.css('div.oturning span.st4 a.reviews::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse_task, meta={'level': level, 'user_id': user_id, 'user_type': user_type})

    