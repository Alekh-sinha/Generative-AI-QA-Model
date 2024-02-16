# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 02:58:23 2024

@author: alekh
"""

class cfg:
    def __init__(self):
        self.path='latest_file' #folder where latest file whose embedding are not yet generated are kept
        self.tokens_per_chunk=256
        self.collection='upsc_2024'
        self.old_file='gs://upsc_2024/old_file' # folder location where files after embedding generation are moved
        self.chroma='gs://upsc_2024/chromadb'#Folder where all chroma collections are stored
        self.model='mistralai/Mistral-7B-Instruct-v0.1'
        self.data='data.json'#Input data file for training
        self.max_length=512
        self.batch_size=1
        self.epochs=1
        self.lr=1e-3
        self.bucket='upsc_2024'
        self.GCS_STAGING = "gs://upsc_2024/pipeline_root/"
        self.MACHINE_TYPE="n1-highmem-8"
        self.REPLICA_COUNT=1
        self.TRAIN_IMAGE_URI='gcr.io/XXXXX/pytorch_training_upsc_2024'
        self.PROJECT_ID='XXXXX'
