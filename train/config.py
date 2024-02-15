# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 02:58:23 2024

@author: alekh
"""

class cfg:
    def __init__(self):
        self.path='latest_file'
        self.tokens_per_chunk=256
        self.collection='upsc_2024'
        self.old_file='gs://upsc_2024/old_file'
        self.chroma='gs://upsc_2024/chromadb'
        self.model='mistralai/Mistral-7B-Instruct-v0.1'
        self.bucket='upsc_2024'
        self.data='data.json'
        self.max_length=512
        self.batch_size=1
        self.epochs=1
        self.lr=1e-3
        self.bucket='upsc_2024'
        self.GCS_STAGING = "gs://upsc_2024/pipeline_root/"
        self.MACHINE_TYPE="n1-highmem-8"
        self.REPLICA_COUNT=1
        self.TRAIN_IMAGE_URI='gcr.io/qualified-abode-411820/pytorch_training_upsc_2024'
        self.PROJECT_ID='qualified-abode-411820'
