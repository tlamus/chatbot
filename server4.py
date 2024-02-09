import os, glob
import sys 
import logging
import asyncio
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json 


from typing import List, Set, Optional, Dict, Tuple, Any, AsyncIterable


from langchain_community.document_loaders import TextLoader, DataFrameLoader, JSONLoader

from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.vectorstores import FAISS, Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ChatMessageHistory

# from elasticsearch import logger as es_logger
# from langchain_community.vectorstores.elasticsearch import logger as lc_logger
from sentence_transformers.SentenceTransformer import logger as st_logger
from langchain_community.llms import LlamaCpp

from langchain.chains import RetrievalQA
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sklearn.metrics.pairwise import cosine_similarity

#turn off the loggers
LOGLEVEL = 50
st_logger.setLevel(LOGLEVEL)

logger = logging.getLogger('example_logger')

# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a file handler which logs even debug messages
fh = logging.FileHandler('server4.log')
logger.addHandler(fh)


from utils import nostdout, get_a_list_of_filepath, create_retriever, create_docs, create_memory, get_keywords



SPLITTER_PARAMS= { "chunk_size" : 100, "chunk_overlap" : 0}

                 
data_dir: str = "/articles",
ext: str = "*.txt",

CONTEXT = None
PROMPT = None
    
HISTORY = ChatMessageHistory()    #elasticseachchathistory()

LLAMA_GUFF = "/Users/tlam/Documents/mistral/saved_models/llama-2-7b-chat.Q8_0.gguf"

CALLBACK_HANDLER = AsyncIteratorCallbackHandler()
CALLBACK_MANAGER = CallbackManager([StreamingStdOutCallbackHandler()])
MODEL_1 = LlamaCpp(
    model_path=LLAMA_GUFF,
    temperature=0,
    max_tokens=256,         
    top_p=1,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
    n_ctx=2048
)

MODEL_2 = LlamaCpp(
    model_path=LLAMA_GUFF,
    temperature=0,
    max_tokens=128,         
    top_p=1,
    verbose=False,
    n_ctx=2048
)

class TemplateManager:
    '''
        These are the prompt templates corresponding to each phase of a conversation.
        Start, continue, change topic, recall topic or end a discusion.
    '''
    def __init__(self):
        self.template=f'''
                            [INST]
                                <<SYS>>
                                    You are a professional assistant.
                                    DO NOT give false information.
                                    Always answer as exactly and shortly as possible,           
                                <</SYS>>
                            [/INST]
                          '''
        
                
    def get_template(self,awareness: str, query: str, context: str) -> str:
        if awareness == "start":
            self.template = self.start_discussion(query, context)
        elif awareness == "continue":
            self.template = self.continue_discussion(query, context)
        elif awareness == "change":
            self.template = self.change_discussion(query,context)
        elif awareness == "recall":
            self.template = self.recall_discussion(query, context)
        
            
        return self.template
        
    def start_discussion(self,query, context) -> None:
        template=f'''
            [INST] <<SYS>>
                      You are a professional assistant.
                      Always answer {query} as exactly and shortly as possible,
                      DO NOT give false information.                      
                   <</SYS>>
                    
            [/INST]
        '''
        return template
        
    def continue_discussion(self,query, context) -> None:
        template=f'''
        [INST]
            <<SYS>>
                You are a professional assistant.
                Analyze this {context} and {query}, answer the user's question as similar to the documents as possible.
                DO NOT give false information.
                Always answer as exactly and shortly as possible,           
            <</SYS>>
        [/INST]
        '''
        return template

    def change_discussion(self,query, context) -> str:
        template=f'''
        [INST]
            <<SYS>>
                You are a professional assistant.
                Silently analyze if this {context} is related to {query}.
                Answer {query} according to the documents as accurately as possible.
                DO NOT give false information.
                Always answer as exactly and shortly as possible,           
            <</SYS>>
            
        [/INST]
        '''
        return template
        
    def recall_discussion(self,query, context) -> str:
        template=f'''
        [INST]
            <<SYS>>
                You are a professional assistant.
                Based on this {context}, answer the user's question as similar to the documents as possible.
                DO NOT give false information.
                Always answer as exactly and shortly as possible,           
            <</SYS>>
            {query}
        [/INST]
        '''
        return template
          
EMBEDDING_HF = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_ST = SentenceTransformer("all-MiniLM-L6-v2")
        
file_list = get_a_list_of_filepath('filepath.txt')
CURRENT_FILE = file_list.split("\n")[0]
FILENAME = CURRENT_FILE.split("/")[-1]
SUB_DOCS = create_docs(CURRENT_FILE)        
STORAGE = create_retriever(SUB_DOCS, EMBEDDING_HF, "CHROMA")

#CREATE A LIST OF THE EXISTING .TXT FILES        
DOCS = create_docs("filepath.txt", **SPLITTER_PARAMS)
FAISS_DB = create_retriever(DOCS, EMBEDDING_HF, "FAISS")

MEMORY_NAME= "Buffer"   # KG or None

MEMORY = create_memory(MEMORY_NAME, llm = MODEL_1)

CHAIN_1 = RetrievalQA.from_chain_type(
        llm = MODEL_1,
        chain_type="stuff",
        retriever= STORAGE.as_retriever(search_kwargs={"k": 1}),
        memory = MEMORY
)

        
CHAIN_2 = RetrievalQA.from_chain_type(
    llm = MODEL_2,
    chain_type="stuff",
    retriever= STORAGE.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=False,
    verbose=False
)

ALREADY_DISCUSSED = [] 

AWARENESS = "start"
QUERY= ""


        
async def analyze_intention(query):
    global AWARENESS
    global CONTEXT
    global FILENAME
    global ALREADY_DISCUSSED
    global QUERY
    
    QUERY = query
    
    if switch_context(QUERY):
        AWARENESS = "change"
        
        
        # await switch_topic()
        
        # CONTEXT = None
        # if FILENAME in ALREADY_DISCUSSED:
        #     AWARENESS = "recall"
        # else:
        #     ALREADY_DISCUSSED.append(FILENAME)
        #     AWARENESS = "change"
        #     CONTEXT = "The user has switched to a new discussion"
    if QUERY.lower() == "bye":
        AWARENESS = "end"
        CONTEXT = "Thank you and Goodbye"
        print(CONTEXT)


# async def generate_response(message: str):
    

#     async for token in callback_handler.aiter():
#         yield token

#     await run
            

async def select_document( data_dir: str, ext: str):
    global FILENAME
    global CURRENT_FILE
    
    file_list = glob.glob(os.path.join(os.getcwd() + data_dir, ext))
    for item in file_list:
        print(item.split("/")[-1])
    try:
        await choose_a_section()
    except Exception as e:
        print(f"Thread 2 Error {e}")
    FILENAME = CURRENT_FILE.split("/")[-1]
    print(f"Q&A: I found {FILENAME} that matches with your search")

async def choose_a_section():
    global AWARENESS
    global CURRENT_FILE
    global QUERY
    
    QUERY = input("User: ")
    if QUERY.lower() == "bye":
        AWARENESS = "end"
    kw = get_keywords(QUERY)
    query = " ".join(kw)
    results = await FAISS_DB.asimilarity_search(query, search_kwargs={"k": 1} )
    
    CURRENT_FILE = results[0].page_content.split("\n")[0]

def switch_context(query, threshold: float=0.5) -> bool:
    '''
        
    '''
    new_query = EMBEDDING_ST.encode([query])
    new_discussion = EMBEDDING_ST.encode(["I want to discuss a new topic"])
    if cosine_similarity(new_query,sample_query)[0][0] > threshold:
        return True
    return False

def (query, threshold: float=0.5) -> bool:
    '''
        
    '''
    new_query = EMBEDDING_ST.encode([query])
    new_discussion = EMBEDDING_ST.encode(["I want to discuss a new topic"])
    if cosine_similarity(new_query,sample_query)[0][0] > threshold:
        return True
    return False

async def switch_topic(data_dir: str="/articles", ext: str="*.txt"):
    global FILENAME
    global CHAIN_1
    global CHAIN_2
    global SUB_DOCS
    global STORAGE
    global QUERY 
    global CURRENT_FILE
    print("Please select the available chapter")
    try:
        await select_document(data_dir, ext)
    except Exception as e:
        print(f"Thread_3 error {e}")
    
    SUB_DOCS = create_docs(CURRENT_FILE)
    STORAGE = create_retriever(SUB_DOCS, EMBEDDING_HF, "CHROMA")
    
    CHAIN_1 = RetrievalQA.from_chain_type(
        llm = MODEL_1,
        chain_type="stuff",
        retriever=STORAGE.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        verbose=False
    )
    
    CHAIN_2 = RetrievalQA.from_chain_type(
        llm = MODEL_2,
        chain_type="stuff",
        retriever=STORAGE.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        verbose=False
    )
    QUERY = f"summarize the main idea of {FILENAME[-4:]} file"
    FILENAME = FILENAME[-4:]

class Message(BaseModel):
    content: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/stream_chat/", response_class=StreamingResponse)
async def stream_chat(message: Message):
    global AWARENESS
    global CONTEXT

    if AWARENESS=="start":
        print("Q&A_Bot: ")
        print("Hello, I'm your assistant. How may I help you today?")
    elif AWARENESS=="end":
        END = f"""[INST]<<SYS>>The User want to terminate the conversation. Respectfully response<</SYS>>[INST]"""
        return StreamingResponse(MODEL_2.invoke(END), media_type="text/event-stream")
        
    logger.info(f"current AWARENESS: {AWARENESS}")
    logger.info(f"current CONTEXT: {CONTEXT}")
    logger.info(f"current MSG: {message.content}")
    await analyze_query(message.content) 

    PROMPT = TemplateManager().get_template(AWARENESS, message.content, CONTEXT)
    
    results = CHAIN_1(PROMPT)
    SUMMARY = f"""[INST]<<SYS>>summarize {results['result']} in one sentence<</SYS>>[INST]"""
    
    CONTEXT = MODEL_2.invoke(SUMMARY)
    
    AWARENESS="continue"
    return StreamingResponse(results["result"], media_type="text/event-stream")

# @app.post("/stream_chat/")
# async def stream_chat(message: Message):
#     prompt_template= f'''[INST] <<SYS>>
#                                         You are a professional assistant. 
#                                         your answers are always less than 250 words.
#                                         Always answer as exactly as possible.
#                                         DO NOT give false information.
                                        
#                                 <</SYS>>
#                                 {message.content}
#                         [/INST]
#                     '''
    
    
#     return StreamingResponse(results["result"], media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=6677, log_level=50)