# backend for Medibot
# import required packages
from flask import Flask, jsonify, request
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain import OpenAI
from langchain import HuggingFaceHub

from flask_cors import CORS
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

app = Flask(__name__)
CORS(app, origins='*')

print("loading environment variables")
load_dotenv()

print("loading chromadb emeddings")
chromadb_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
print("loading langchain emeddings")
langchain_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")