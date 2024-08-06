import os

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def create_vectorstore(
    sources, persist_directory, embedding=OpenAIEmbeddings()
) -> Chroma:
    """
    create_vectorstore of documentation

    Args:
        sources (str): path to documentation
        persist_directory (_type_): _description_
        embedding (_type_, optional): _description_. Defaults to OpenAIEmbeddings().

    Returns:
        _type_: _description_
    """
    documents = []

    for source in sources:
        if isinstance(source, str):
            if os.path.isdir(source):
                if source.endswith(".txt"):
                    loader = DirectoryLoader(
                        source, glob="*.txt", loader_cls=TextLoader
                    )
                documents.extend(loader.load())
            elif os.path.isfile(source):
                loader = TextLoader(source)
                documents.extend(loader.load())
        elif isinstance(source, list):
            documents.extend(source)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory,
    )

    return vectordb


base_dir = "{base_dir}"
sources = [
    f"{base_dir}api_documentation/mygene/raw_doc_mygene.txt",
    f"{base_dir}general_bioinformatics",
    f"{base_dir}api_documentation/aniseed/aniseed_api_doc.txt",
]

persist_directory = "/home/data/persistant_files/vectorstores/general_datastore/"
vectordb = create_vectorstore(sources, persist_directory)
