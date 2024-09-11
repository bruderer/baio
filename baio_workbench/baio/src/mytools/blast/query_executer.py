import re
import time
from urllib.parse import urlencode

import requests

from baio.src.non_llm_tools import log_question_uuid_json

from . import BlastQueryRequest


def submit_blast_query(request_data: BlastQueryRequest):
    """FIRST function to be called for each BLAST query.
    It submits the structured BlastQueryRequest obj and return the RID.
    """
    data = {
        "CMD": request_data.cmd,
        "PROGRAM": request_data.program,
        "DATABASE": request_data.database,
        "QUERY": request_data.query,
        "FORMAT_TYPE": request_data.format_type,
        "MEGABLAST": request_data.megablast,
        "HITLIST_SIZE": request_data.max_hits,
    }
    # Include any other_params if provided
    if request_data.other_params:
        data.update(request_data.other_params)
    # Make the API call
    query_string = urlencode(data)
    # Combine base URL with the query string
    full_url = f"{request_data.url}?{query_string}"
    # Print the full URL
    request_data.full_url = full_url
    print("Full URL built by retriever:\n", request_data.full_url)
    response = requests.post(request_data.url, data=data)
    response.raise_for_status()
    # Extract RID from response
    match = re.search(r"RID = (\w+)", response.text)
    if match:
        return match.group(1)
    else:
        raise ValueError("RID not found in BLAST submission response.")


def fetch_and_save_blast_results(
    request_data: BlastQueryRequest,
    blast_query_return: str,
    save_path: str,
    question: str,
    log_file_path: str,
    wait_time: int = 15,
    max_attempts: int = 10000,
):
    """SECOND function to be called for a BLAST query.
    Will look for the RID to fetch the data
    """
    file_name = f"BLAST_results_{request_data.question_uuid}.txt"
    if request_data.question_uuid is not None and request_data.full_url is not None:
        log_question_uuid_json(
            request_data.question_uuid,
            question,
            file_name,
            save_path,
            log_file_path,
            request_data.full_url,
            tool="BLAST",
        )
    else:
        print("Warning: question_uuid is None, skipping logging")
    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    check_status_params = {
        "CMD": "Get",
        "FORMAT_OBJECT": "SearchInfo",
        "RID": blast_query_return,
    }
    get_results_params = {"CMD": "Get", "FORMAT_TYPE": "XML", "RID": blast_query_return}
    # Check the status of the BLAST job
    for attempt in range(max_attempts):
        status_response = requests.get(base_url, params=check_status_params)
        status_response.raise_for_status()
        status_text = status_response.text
        if "Status=WAITING" in status_text:
            print(f"{request_data.question_uuid} results not ready, waiting...")
            time.sleep(wait_time)
        elif "Status=FAILED" in status_text:
            with open(f"{save_path}{file_name}", "w") as file:
                file.write("BLAST query FAILED.")
        elif "Status=UNKNOWN" in status_text:
            with open(f"{save_path}{file_name}", "w") as file:
                file.write("BLAST query expired or does not exist.")
            raise
        elif "Status=READY" in status_text:
            if "ThereAreHits=yes" in status_text:
                print(
                    "{request_data.question_uuid} results are ready, retrieving and "
                    "saving..."
                )
                results_response = requests.get(base_url, params=get_results_params)
                results_response.raise_for_status()
                # Save the results to a file
                print(f"{save_path}{file_name}")
                with open(f"{save_path}{file_name}", "w") as file:
                    file.write(results_response.text)
                print(
                    f"Results saved in BLAST_results_{request_data.question_uuid}.txt"
                )
                break
            else:
                with open(f"{save_path}{file_name}", "w") as file:
                    file.write("No hits found")
                break
        else:
            print("Unknown status")
            with open(f"{save_path}{file_name}", "w") as file:
                file.write("Unknown status")
            break
    if attempt == max_attempts - 1:
        raise TimeoutError("Maximum attempts reached. Results may not be ready.")
    return file_name
