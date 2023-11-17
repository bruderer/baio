from langchain.vectorstores import Chroma

db = Chroma(persist_directory="persistant_vectorstore/vectorstores/")
aniseed_db = Chroma(persist_directory="persistant_vectorstore/vectorstores/aniseed_datastore")