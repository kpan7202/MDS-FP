# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:05:21 2017

@author: Kurniawan
"""

import html.parser as parser

class TagStripper(parser.HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
        
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags_spaces(txt):
    s = TagStripper()
    s.feed(txt)
    return ' '.join(s.get_data().split())
