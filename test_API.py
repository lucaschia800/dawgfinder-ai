from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings 
import openai 
from CustomNLSQLRetriever import CustomNLSQLRetriever
import os
from sqlalchemy import (
    create_engine
)
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import text
from llama_index.core import PromptTemplate






pinecone_api_key = os.environ.get("pinEcone_API")
if pinecone_api_key is None:
    raise ValueError("Please set the environment variable pinecone_API")
pc = Pinecone(api_key=pinecone_api_key)