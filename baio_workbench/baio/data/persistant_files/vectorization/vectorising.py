import os
from data.persistant_files import constants
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from typing import List
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.document_loaders import TextLoader


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# #from data.constants import constants 
os.environ["OPENAI_API_KEY"] = constants.APIKEY
persist_directory = '/home/persistant_vectorstore/vectorstores/'

embedding = OpenAIEmbeddings()

#####
#####       MYGENE DOC
#####

from langchain.document_loaders import TextLoader

#we load the api data
loader = TextLoader("/home/persistant_vectorstore/data/api_documentation/mygene/raw_doc_mygene.txt")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
all_splits = text_splitter.split_documents(data)


####
####         NCBI & BIOPYTHON
####



loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)

loader_pdf = DirectoryLoader('/home/persistant_vectorstore/user_manuals/', 
                             glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader_pdf.load()

#next up is making the index
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

#now we splitted the documents into chunk sizes of 500 with 100 overlapping
docs = text_splitter.split_documents(documents)



####
####        ANISEED
####

#we load the api data
loader = TextLoader("/home/persistant_vectorstore/user_manuals/aniseed_api_doc.txt")
anisseed_data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
anisseed_data_all_splits = text_splitter.split_documents(anisseed_data)

persist_directory_aniseed = '/home/persistant_vectorstore/vectorstores/aniseed_datastore/'
vectordb_anissed = Chroma.from_documents(documents=anisseed_data_all_splits,
                                            embedding=embedding,
                                            persist_directory=persist_directory_aniseed)




####
####        STORING ALL SPLITS
####
# Store 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectordb = Chroma.from_documents(documents=all_splits+docs + anisseed_data_all_splits,
                                            embedding=embedding,
                                            persist_directory=persist_directory)

