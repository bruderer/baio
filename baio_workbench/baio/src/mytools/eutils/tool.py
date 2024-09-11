import os
import uuid

from baio.src.mytools.utils import load_vector_store
from baio.src.non_llm_tools.utilities import log_question_uuid_json

from . import (
    eutils_API_query_generator,
    handle_non_snp_query,
    handle_snp_query,
    result_file_extractor,
    save_response,
)


def eutils_tool(question: str, llm, embedding):
    """Tool to make any eutils query, creates query, executes it, saves result in file
    and reads answer"""
    print("Running: Eutils tool")
    log_file_path = "./baio/data/output/eutils/results/log_file/eutils_log.json"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_path = "./baio/data/output/eutils/results/files/"
    path_vdb = "./baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index"
    doc = load_vector_store(path_vdb, embedding)

    # Check if the directories for file_path and log_file_path exist, if not, create
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print("1: Building API call for user question:")
    # FIRST API CALL TO GET IDs
    efetch_response_list = []
    api_call = eutils_API_query_generator(question, llm, doc)
    api_call.question_uuid = str(uuid.uuid4())

    print(f"2: Executing API call: {api_call.full_search_url}")
    if api_call.db != "snp":
        efetch_response_list = handle_non_snp_query(api_call)
    else:
        efetch_response_list = handle_snp_query(api_call)
    result_file_name = save_response(
        efetch_response_list, file_path, api_call.question_uuid
    )
    log_question_uuid_json(
        api_call.question_uuid,
        question,
        result_file_name,
        file_path,
        log_file_path,
        api_call.full_search_url,
        tool="eutils",
    )
    result = result_file_extractor(
        question, api_call.question_uuid, log_file_path, llm, embedding
    )

    return result
