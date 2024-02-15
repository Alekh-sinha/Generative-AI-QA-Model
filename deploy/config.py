# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 02:58:23 2024

@author: alekh
"""

class cfg:
    def __init__(self):
        self.path='gs://upsc_2024/latest_file'
        self.tokens_per_chunk=256
        self.collection='upsc_2024'
        self.old_file='gs://upsc_2024/old_file'
        self.chroma='gs://upsc_2024/chromadb'
        self.model='mistralai/Mistral-7B-Instruct-v0.1'
