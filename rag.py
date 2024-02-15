# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:35:54 2024

@author: alekh
"""
import requests
from transformers import MarkupLMFeatureExtractor
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from config import cfg
import subprocess
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
""
CFG=cfg()
""
def extract_csv(file):
    df=pd.read_csv(file)
    return ('\n\n'.join(list(df['text'].fillna(''))))
""
def extract_pdf(file):
    reader = PdfReader(file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return ('\n\n'.join(pdf_texts))
def text_extraction(file_name):
    text=[]
    for file in file_name:
        if file.split('.')[-1]=='csv':
            text.append(extract_csv(file))
        if file.split('.')[-1]=='pdf':
            text.append(extract_pdf(file))
    return ('\n\n'.join(text))
""
class embedding:
    def __init__(self,cfg):
        self.character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""],chunk_size=1000,chunk_overlap=0)
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path="deploy/chromadb")
        self.token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=cfg.tokens_per_chunk)
    def collection_update(self,cfg,text):
        self.character_split_texts = self.character_splitter.split_text(text)
        token_split_texts = []
        for text in self.character_split_texts:
            token_split_texts += self.token_splitter.split_text(text)
        self.chroma_collection = self.chroma_client.get_or_create_collection(cfg.collection, embedding_function=self.embedding_function)
        l=self.chroma_collection.count()
        print(l)
        ids = [str(i+l) for i in range(len(token_split_texts))]
        self.chroma_collection.add(ids=ids, documents=token_split_texts)
""
def embedding_generation(CFG):
    ###Supported format is csv,
    #try:
    #    os.mkdir('chromadb')
    #except:
    #    print('chromadb is already present')
    #cmd='gsutil cp -r {}/* chromadb'.format(CFG.chroma)
    #subprocess.run(cmd.split(' '))
    ###########################################
    #try:
    #    os.mkdir(CFG.path.split('/')[-1])
    #except:
    #    print('{} is already present'.format(CFG.path.split('/')[-1]))
    #cmd='gsutil cp -r {}/* {}'.format(CFG.path,CFG.path.split('/')[-1])
    #subprocess.run(cmd.split(' '))
    file_name_path=[]
    for root, dirs_list, files_list in os.walk('workingdir'):
        for file_name in files_list:
            if (os.path.splitext(file_name)[-1] == '.csv')|(os.path.splitext(file_name)[-1] == '.pdf'):
                file_name_path.append(os.path.join(root, file_name))
    ##################################
    text=text_extraction(file_name_path)
    emb=embedding(CFG)
    emb.collection_update(CFG,text)
    #############################################
    #cmd='gsutil mv {}/* {}'.format(CFG.path,CFG.old_file)
    #subprocess.run(cmd.split(' '))
    #########################################
    #cmd='gsutil cp -r chromadb/* {}'.format(CFG.chroma)
    #subprocess.run(cmd.split(' '))
    #for i in file_name_path:
    #    cmd='gsutuil cp {}/* {}'.format(CFG.path,CFG.old_file)
    #    subprocess.run(cmd.split(' '))
        #shutil.move(i,i.replace(CFG.path,CFG.old_file))
""
def main(CFG):
    embedding_generation(CFG)
if __name__ == "__main__":
    main(CFG)
