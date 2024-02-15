# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 03:13:18 2024

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
from lxml import etree
from google.cloud import storage
""
CFG=cfg()
def xpath_extraction(URL):
    page = requests.get(URL)
    parser = etree.HTMLParser()
    root = etree.fromstring(page.text,parser)
    tree = etree.ElementTree(root)
    xpaths=[]
    for e in root.iter():
        xpaths.append(tree.getpath(e))
    return xpaths,page,tree
""
def string_score(string1,string2):
  set1=set(string1.split('/'))
  set2=set(string2.split('/'))
  score=len(set1.intersection(set2))/len(set1.union(set2))
  return(score)
""
def xpaths_mod(df,xpaths=[]):
  score=[string_score(df['xpath'],xpath) for xpath in xpaths]
  ind=np.argmax(np.array(score))
  return xpaths[ind]
""
def filter(df):
  return((len(df['nodes'].split(' '))>3)&(df['xpath'].split('/')[-1]=='a'))*1
""
def fetch_data(URL):
    #print(URL)
    page = requests.get(URL)
    feature_extractor = MarkupLMFeatureExtractor()
    encoding = feature_extractor(page.text)
    ################################################
    df=pd.DataFrame(encoding['xpaths'][0],columns=['xpath'])
    df['nodes']=encoding['nodes'][0]
    df['filter']=[((i.split('/')[-1].find('p')>=0)&(i.split('/')[-1].find('span')<0)&(i.split('/')[-1].find('script')<0))*1 for i in df['xpath']]
    df=df[df['filter']==1]
    return (' '.join(list(df['nodes'])))
""
def preprocess(df):
  if len(df['url'])==0:
    return ''
  else:
    return fetch_data(df['url'][0])
""
def main():
    try:
        os.mkdir('workingdir')
    except:
        print('chromadb is already present')
    URL = "https://write you own.com/"
    xpaths,page,tree=xpath_extraction(URL)
    feature_extractor = MarkupLMFeatureExtractor()
    # single example
    encoding = feature_extractor(page.text)
    df=pd.DataFrame(encoding['xpaths'][0],columns=['xpath'])
    df['nodes']=encoding['nodes'][0]
    ##############################################
    df['xpaths_mod']=df.apply(xpaths_mod,axis=1,xpaths=xpaths)
    df['filter']=df.apply(filter,axis=1)
    df=df[df['filter']==1]
    df['url']=[tree.xpath(i+'/@href') for i in df['xpaths_mod']]
    df['text']=df.apply(preprocess,axis=1)
    filename='pre_processed_hindu_{}.csv'.format(str(datetime.now()).split(' ')[0])
    df.to_csv(filename,index=None)
    ##############################################
    client = storage.Client()
    bucket = client.get_bucket(CFG.bucket)
    bucket.blob(CFG.path+'/'+filename).upload_from_string(df.to_csv(), 'text/csv')
    #####################################################
if __name__ == "__main__":
    main()
