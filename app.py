# backend for Medibot
# import required packages
from flask import Flask, jsonify, request
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from langchain import OpenAI
from langchain import HuggingFaceHub

from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, origins='*')

print("loading environment variables")
load_dotenv()