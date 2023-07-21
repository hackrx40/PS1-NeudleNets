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
import os
from dotenv import load_dotenv, find_dotenv


app = Flask(__name__)
CORS(app, origins='*')

print("loading environment variables...")
_ = load_dotenv(find_dotenv()) # read local .env file

print("loading chromadb emeddings...")
chromadb_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

print("loading langchain emeddings...")
langchain_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("loading AWS deployed ChromaDB Stack environment variables...")
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = os.getenv('CHROMA_PORT')
COLLECTION_NAME=os.getenv('COLLECTION_NAME')

# Setup chroma settings
chroma_client_settings = Settings(
        chroma_api_impl="rest",
        chroma_server_host=CHROMA_HOST,
        chroma_server_http_port=CHROMA_PORT,
    )

# Set up chroma client settings
chroma_client = chromadb.Client(settings=chroma_client_settings)
print("ChromaDB's heartbeat: ", chroma_client.heartbeat())

# Set up Chroma collection
collection = chroma_client.get_collection(name=COLLECTION_NAME,
                                        embedding_function=chromadb_embeddings)
print("collection counts: ", collection.count())

# Set up langchain vectorDB
vectordb = Chroma(client=chroma_client,client_settings=chroma_client_settings,
                  collection_name=COLLECTION_NAME,
                  embedding_function=langchain_embeddings)
print("set up vectordb")

class Session:
    """Template class for a session with the chatbot"""
    def __init__(self):
        self.chroma_client = chroma_client
        self.collection = collection
        self.vectordb = vectordb

class User_Session(Session):
    """Class to manage one user session on the chatbot platform"""
    def __init__(self):
        self.selected_llm ="OpenAI"
        self.query = ""
        self.sources = []
        self.chat_history = []
        self.responses = []
        self.source_ids = []
        self.source_cites = []
        self.rating = 5 # initalise every content with 5 star

    def get_llm(self):
        """Load the LLM from langchain; default LLM is OpenAI"""
        if self.selected_llm == "Google-Flan-T5-XXL":
            llm = HuggingFaceHub(repo_id='google/flan-t5-xxl', model_kwargs={"temperature":0.1, "max_new_tokens":1180})
        elif self.selected_llm == "OpenAssistant":
            llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b", model_kwargs={"temperature":0.1, "max_new_tokens":768})
        elif self.selected_llm == "Falcon-7b-instruct":
            llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1, "max_new_tokens":256})
        else:
            llm = OpenAI()
        return llm

    def response_from_llm(self):
        """Generate response from the LLM loaded"""
        llm = self.get_llm()
        self.response = ""
        self.source_id = ""
        self.source_cites = []
        self.chat_history = []
        
        # First we get the best response out of all the sources based on the
        # selected sources
        # We need to create the search_kwargs filter
        # We create the or_filter from all the sources
        or_filters =[]
        for source in self.sources:
            source_dict = {'source':source}
            or_filters.append(source_dict)
        
        # Define search kwargs
        search_kwargs = {
                'k':1,
                'filter':{
                    "$and": [
                        {
                            "$or": or_filters
                        },
                        {
                            "rating": {'$gt': 4}
                        }
                    ]
                }
            }

        # Define retrieval chain
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
        
        # Get the document with
        # ids == self.source_ids[0] :: this returns a dictionary
        doc = self.vectordb.get(ids=[self.source_id])
        
        # Get the current rating of the article
        curr_rating = doc['metadatas'][0]['rating']

        # Update the rating
        new_rating = (curr_rating + self.rating) / 2
        doc['metadatas'][0]['rating'] = new_rating
        
        # Create a langchain document from dictionary
        document =Document(
            ids=doc['ids'][0],
            page_content=doc['documents'][0],
            metadata={
                'source': doc['metadatas'][0]['source'],
                'url': doc['metadatas'][0]['url'],
                'disease': doc['metadatas'][0]['disease'],
                'rating': doc['metadatas'][0]['rating'],
            }
        )

        # Update the vectordb
        self.vectordb.update_document(self.source_id, document)

class Admin_Session(Session):
    """Child class to manage Admin Sessions on chatbot"""
    def __init__(self):
        Session.__init__(self)
        self.file = None
        self.fileName = ""
        self.fileContent = ""

    def create_document(self):
        """Convert file to document that can be added to ChromaDB"""
        # print("Inside the function")
        print(self.fileContent)

        # Apply in the ChromaDB's Schema
        document = Document(
            page_content = self.fileContent,
            metadata = {
                'source': self.fileName,
                'url': None,
                'disease': 'Generic',
                'rating': 5
            }
        )
        # print("This is the document: ", document)

        # Chunk the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20
        )

        documents = text_splitter.split_documents([document])
        print("chunks of documents created successfully")
        
        return documents
    
    def add_document_to_vectordb(self, doc):
        """Add document to ChromaDB"""  
        print("inside adding the document function")
        print(doc)
        print(doc.metadata)
        print(type(doc.metadata))
        self.collection.add(
            ids=str(collection.count()+1),
            documents=doc.page_content,
            metadatas = doc.metadata
        )
        print("The count of the collection in ChromaDB: ", collection.count())

