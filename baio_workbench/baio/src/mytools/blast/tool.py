import os

from baio.src.mytools.utils import load_vector_store

from . import (
    BLAST_answer,
    BLAST_api_query_generator,
    fetch_and_save_blast_results,
    submit_blast_query,
)


def blast_tool(question: str, llm, embedding):
    """BLAST TOOL, use this tool if you need to blast a dna sequence on the blast data
    base on ncbi"""
    log_file_path = "./baio/data/output/BLAST/logfile.json"
    save_file_path = "./baio/data/output/BLAST/files/"
    path_vdb = "./baio/data/persistant_files/vectorstores/BLAST_db_faiss_index"
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # generate api call
    doc = load_vector_store(path_vdb, embedding)
    query_request = BLAST_api_query_generator(question, doc, llm)
    print(query_request)
    current_uuid = query_request.question_uuid  # Get the UUID of the current request
    # submit BLAST query
    rid = submit_blast_query(query_request)
    # retrieve BLAST results
    fetch_and_save_blast_results(
        query_request, rid, save_file_path, question, log_file_path
    )

    result = BLAST_answer(log_file_path, question, current_uuid, 200, embedding, llm)

    return result["answer"]
