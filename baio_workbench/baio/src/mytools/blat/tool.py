import os

from baio.src.mytools.utils import load_vector_store
from baio.src.non_llm_tools.utilities import log_question_uuid_json

from . import (
    BLAT_answer,
    BLAT_API_call_executor,
    BLAT_api_query_generator,
    save_BLAT_result,
)


def BLAT_tool(question: str, llm, embedding):
    """BLAT TOOL, use this tool if you need to BLAT a dna sequence on the BLAT data base
    on ncbi"""
    print("Executing: BLAT Tool")
    log_file_path = "./baio/data/output/BLAT/logfile.json"
    file_path = "./baio/data/output/BLAT/files/"
    path_vdb = "./baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Ensure the files are created empty
    try:
        open(file_path, "x").close()  # Create an empty file if it does not exist
    except FileExistsError:
        print(f"{file_path} already exists")

    try:
        open(
            log_file_path, "x"
        ).close()  # Create an empty log file if it does not exist
    except FileExistsError:
        print(f"{log_file_path} already exists")

    # generate api call
    doc = load_vector_store(path_vdb, embedding)
    query_request = BLAT_api_query_generator(question, llm, doc)
    # print(query_request)
    BLAT_response = BLAT_API_call_executor(query_request)
    # print(BLAT_response)
    file_name, full_file_path = save_BLAT_result(
        query_request, BLAT_response, file_path
    )

    result = BLAT_answer(full_file_path, question, llm, embedding)
    print("BLAT_tool before saving log")
    print(result)
    log_question_uuid_json(
        query_request.question_uuid,
        question,
        file_name,
        file_path,
        log_file_path,
        query_request.full_url,
        answer=result,
        tool="BLAT",
    )

    return result
