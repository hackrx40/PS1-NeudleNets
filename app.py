# Backend for Medibot
# Flask imports
from flask import Flask, jsonify, request
from flask_cors import CORS

# Langchain imports
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain import OpenAI
from langchain import HuggingFaceHub

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Other imports
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app, origins='*')

print("loading environment variables...")
load_dotenv()

print("loading chromadb emeddings...")
chromadb_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
print("loading langchain emeddings...")
langchain_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

class User_Session():
    """Class to manage one user session on the chatbot platform"""
    def __init__(self):
        self.selected_llm ="OpenAI"
        self.query =""
        self.sources =[]
        self.chat_history = []
        self.responses=[]
        self.source_ids=[]
        self.source_cites=[]
        self.rating=5

    def get_llm(self):
        """Load the LLM from langchain"""
        llm = OpenAI()
        return llm


    def response_from_llm(self):
        """Generate response from the LLM loaded"""
        llm = self.get_llm()
        self.response =""
        self.source_id = ""
        self.source_cites = []
        self.chat_history=[]
        
        
        qa = ConversationalRetrievalChain.from_llm(llm = llm,
                                    retriever = self.vectordb.as_retriever(),
                                    return_source_documents = True)

        # Get the response
        result = qa({'question':self.query, "chat_history":self.chat_history})
        self.response = result['answer']
        target_document = result['source_documents'][0]
        print(target_document.metadata['source'])
        # Self.source_cites.append(target_document.metadata['url'])
        self.source_id = self.collection.query(query_texts=target_document.page_content,
                                               n_results=1)['ids'][0][0]
        print(len(self.vectordb.get(where={"rating": {'$gt': 4}})['ids']))

        for source in self.sources:
            document = self.vectordb.similarity_search(self.query,k=1 , filter={
                'source':source
            })
            self.source_cites.append(document[0].metadata['url'])
        
        return self.response, self.source_cites
    
    def update_rating(self):
        """Update the rating of a particular doc retrieved"""
        # Get the document with ids == self.source_ids[0] :: this returns a dictionary
        doc = self.vectordb.get(ids=[self.source_id])
        
        # Get the current rating of the article
        curr_rating =doc['metadatas'][0]['rating']

        # Update the rating
        new_rating = (curr_rating+self.rating)/2
        doc['metadatas'][0]['rating'] = new_rating
        
        # Create a langchain document from dictionary
        document =Document(
            ids=doc['ids'][0],
            page_content=doc['documents'][0],
            metadata={
                'source':doc['metadatas'][0]['source'],
                'url':doc['metadatas'][0]['url'],
                'disease':doc['metadatas'][0]['disease'],
                'rating':doc['metadatas'][0]['rating'],
            }
        )

        # Update the vectordb
        self.vectordb.update_document(self.source_id,document)