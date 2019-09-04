# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:20:50 2017

@author: Kurniawan

A class to crawl Witmart.com task details page
"""

import scrapy
from utilities.util import strip_tags_spaces
from utilities.witmart import WitMartJobs, WitMartConnection


class WitmartJobsSpider(scrapy.Spider):
    name = 'witmart-jobs'
    start_urls = ['http://www.witmart.com/logo-design/jobs?s=3'] #url to all logo design completed jobs
    
    # default method to handle parsing the urls stored in start_urls
    def parse(self, response):
        for job in response.css('ul#joblist li'):   
            # for each url if it's not private job (must sign in to see) then follow the link
            link = job.css('a.fromtitle::attr(href)').extract_first()
            if link is not None and link.find('user/sign/in') == -1:
                yield response.follow(link, callback=self.parse_job)

        # parse the next page link if it exists
        next_page = response.css('span.st4 a::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    # method to parse job detail page
    def parse_job(self, response):               
        job_id = response.css('input#jobid::attr(value)').extract_first()
        jobs = WitMartJobs()
        # if the page contains job id that can be parsed and the job_id is not found in collection "jobs"
        if job_id is not None: 
            job_data = jobs.find_job_by_id(job_id)
            if  job_data is None:
                data = {}
                data['job_id'] = job_id
                data['title'] = response.css('div.gj_title h2::text').extract_first().strip()
                
                temp = response.css('div.gj_title p.dq_nav a')       
                data['employer'] = temp[0].css('a::attr(href)').extract_first()[3:]
                data['category'] = temp[1].css('a::text').extract_first().strip()
                data['type'] = temp[2].css('a::text').extract_first().strip()
                
                temp = response.css('table.t_details tr') 
                data['status'] = strip_tags_spaces(temp[0].css('td')[0].css('td').extract_first())[12:] # remove 'Job Status: '
                
                #if the element contains word "reward/Reward"
                if temp[1].css('td')[0].css('b.g-f14::text').extract_first().find('eward') >= 0:
                    data['reward'] = strip_tags_spaces(temp[1].css('td')[0].css('td strong::text').extract_first())
                else: # if there is another detail about the pay rate else set default to Negotiable
                    reward = 'Negotiable'
                    if len(temp) >= 3:
                        reward = strip_tags_spaces(temp[2].css('td').extract_first())
                        idx = reward.find(':')
                        if idx >= 0:
                            reward = reward[idx + 2:].strip()
                    data['reward'] = reward
                
                data['bid_start'] = strip_tags_spaces(temp[0].css('td')[1].css('td').extract_first())[17:] #remove 'Bidding Started: '
                data['bid_end'] = strip_tags_spaces(temp[1].css('td')[1].css('td').extract_first())[15:] #remove 'Bidding Ended: '
                
                # trying to parse the description. Some pages have different element format
                if response.css('div#j-langdes').extract_first() is not None:
                    temp = strip_tags_spaces(response.css('div#j-langdes').extract_first()).strip()
                    if temp == "" and response.css('div#j-hidefortrans').extract_first() is not None:
                        attrs = response.css('div#j-hidefortrans h5::text').extract()
                        values = response.css('div#j-hidefortrans div.JOBDESC').extract()
                        for kvp in zip(attrs,values):
                            temp += kvp[0] + ": " + strip_tags_spaces(kvp[1]) + "; "           
                    data['description'] = temp
                 
                # parse bidders and who won the bid / got hired    
                data['bid_list'] = []
                data['winner_list'] = []
                for bidder in response.css('div#all_bids dl.list'):
                    user_id = bidder.css('dd.col1::attr(value)').extract_first()
                    data['bid_list'].append(user_id)
                    winner = False
                    if bidder.css('dd.zb').extract_first() is not None:
                        data['winner_list'].append(user_id)
                        winner = True
                    # insert / update connection collection
                    self.create_connection(data['job_id'], data['employer'], user_id, winner)
                
                # parse skills required to do the task
                data['required_skills'] = []
                for skill in response.css('div.t_des div.mt20 a'):
                    data['required_skills'].append(skill.css('a::text').extract_first())
    
                # insert data to collection "jobs"                            
                jobs.insert_job(data)
                # if total bidders are > 10 then follow the next page to crawl next bidders
                next_bid_page = response.css('i.next a::attr(href)').extract_first()
                if next_bid_page is not None:
                    yield response.follow(next_bid_page, callback=self.parse_bid)
            else:
                for bidder in job_data['bid_list']:
                    #add logo design relationship
                    winner = False
                    if bidder in job_data['winner_list']:
                        winner = True
                    self.create_connection(job_data['job_id'], job_data['employer'], bidder, winner)
        jobs.close()
        
    # method to parse bidders on page 2 and above
    def parse_bid(self, response):
        job_id = response.css('input#jobid::attr(value)').extract_first()
        jobs = WitMartJobs()
        # find job detail from collection and update bidder list and winner list
        data = jobs.find_job_by_id(job_id)
        if data is not None:
            for bidder in response.css('div#all_bids dl.list'):
                user_id = bidder.css('dd.col1::attr(value)').extract_first()
                if user_id not in data['bid_list']:
                    data['bid_list'].append(user_id)
                winner = False
                if bidder.css('dd.zb').extract_first() is not None and user_id not in data['winner_list']:
                    data['winner_list'].append(user_id)
                    winner = True
                # insert / update connection collection
                self.create_connection(data['job_id'], data['employer'], user_id, winner)
                
            jobs.update_job(data)
         
            next_bid_page = response.css('i.next a::attr(href)').extract_first()
            if next_bid_page is not None:
                yield response.follow(next_bid_page, callback=self.parse_bid)
                
        jobs.close() 

    # method to insert / update connection between employer and worker    
    def create_connection(self, job_id, poster_id, bidder_id, winner = False):    
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
        
        
