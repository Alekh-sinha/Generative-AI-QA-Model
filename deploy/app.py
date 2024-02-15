# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:19:13 2024

@author: alekh
"""

from pydantic import BaseModel

import os
import shutil
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import cfg
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
""
app = FastAPI(title="Deploying FastAPI Apps on GCP")
""
CFG=cfg()
#################################
# ########################################
class Generate(BaseModel):
    text:str

class infer:
    def __init__(self,cfg):
        self.chroma_client = chromadb.PersistentClient(path="chromadb")
        self.chroma_collection = self.chroma_client.get_collection(cfg.collection)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model,low_cpu_mem_usage=True)
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.langchain_chroma = Chroma(client=self.chroma_client,collection_name=cfg.collection,embedding_function=self.embedding_function)
        self.retriever = self.langchain_chroma.as_retriever(search_type="mmr")
    def generate_text(self,query):
        #results = self.chroma_collection.query(query_texts=[query], n_results=3)
        #self.retrieved_documents = results['documents'][0]
        self.retrieved_documents = self.retriever.get_relevant_documents(query,n_results=5)
        self.retrieved_documents=[i.page_content for i in self.retrieved_documents]
        self.information = "\n\n".join(self.retrieved_documents)
        prompt_template = """
        Answer the question based on the context below. Check if context is relevent or related to the question. If it is not relevant return answer as "not sure about the answer"
        question: {question}
        context: {context}
        answer:
        """
        message=prompt_template.format(question=query,context=self.information)
        inputs = self.tokenizer(message, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256,do_sample=True,temperature=0.5)
        text=self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return Generate(text=text)
        #return Generate(text=query)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
infer=infer(CFG)

@app.get("/", tags=["Home"])
def api_home():
    return {'detail': 'Welcome to FastAPI TextGen Tutorial!'}

@app.post("/api/generate", summary="Generate text from prompt", tags=["Generate"], response_model=Generate)
def inference(input_prompt: str):
    return infer.generate_text(input_prompt)
