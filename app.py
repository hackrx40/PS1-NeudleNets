# backend for Medibot
# import required packages
from flask import Flask, jsonify, request
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from langchain import OpenAI
from langchain import HuggingFaceHub