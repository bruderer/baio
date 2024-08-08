import json
import os
import urllib
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Union
from urllib.parse import urlencode

from . import EfetchRequest, EutilsAPIRequest


def execute_eutils_api_call(request_data: Union[EutilsAPIRequest, EfetchRequest]):
    """Define"""
    print("In API caller function\n--------------------")
    print(request_data)
    # Default values for optional fields
    default_headers = {"Content-Type": "application/json"}
    default_method = "GET"

    if isinstance(request_data, EfetchRequest):
        if request_data.db == "gene":
            print("FETCHING")
            # print(request_data)
            if isinstance(request_data.id, list):
                id_s = ",".join(
                    map(str, request_data.id)
                )  # Convert each element to string and join
            else:
                id_s = str(request_data.id)
            query_params = request_data.dict(include={"db", "retmax", "retmode"})
            encoded_query_string = urlencode(query_params)
            query_string = f"{encoded_query_string}&id={id_s}"
            request_data.full_search_url = f"{request_data.url}?{query_string}"

        if request_data.db == "omim":
            print("FETCHIN omim results")
            if isinstance(request_data.id, list):
                id_s = ",".join(
                    map(str, request_data.id)
                )  # Convert each element to string and join
            else:
                id_s = str(request_data.id)
            query_params = request_data.dict(include={"db", "retmax", "retmode"})
            encoded_query_string = urlencode(query_params)
            if request_data.id != "" and request_data.id is not None:
                request_data.url = (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                )
            query_string = f"{encoded_query_string}&id={id_s}"
            request_data.full_search_url = f"{request_data.url}?{query_string}"
    print(f"Requesting x: {request_data.full_search_url}")

    req = urllib.request.Request(
        request_data.full_search_url,  # type: ignore
        headers=default_headers,
        method=default_method,
    )
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read()

            if request_data.retmode.lower() == "json":
                try:
                    return json.loads(response_data)
                except json.JSONDecodeError:
                    print("Warning: Expected JSON, but received non-JSON data.")
                    return response_data.decode("utf-8")
            else:
                return response_data.decode("utf-8")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        if e.code == 429:  # Too Many Requests
            print("Consider implementing rate limiting or exponential backoff.")
        elif e.code >= 500:
            print("Server error. The issue might be temporary.")
        return f"HTTP Error: {e.code} - {e.reason}"

    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return f"URL Error: {e.reason}"

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return f"Unexpected error: {str(e)}"


def extract_id_list(response, retmode: str) -> List[int | str]:
    if retmode == "json":
        return response.get("esearchresult", {}).get("idlist", [])
    elif retmode == "xml":
        return response.get("eSearchResult", {}).get("IdList", []).get("Id", [])
    return []


def make_efetch_request(
    api_call: EutilsAPIRequest, id_list: List[int | str]
) -> EfetchRequest:
    return EfetchRequest(db=api_call.db, id=id_list, retmode=api_call.retmode)


def handle_non_snp_query(api_call: EutilsAPIRequest) -> List[Union[dict, str]]:
    response = execute_eutils_api_call(api_call)
    id_list = extract_id_list(response, api_call.retmode)
    efetch_request = make_efetch_request(api_call, id_list)
    efetch_response = execute_eutils_api_call(efetch_request)
    return [efetch_response]


def handle_snp_query(api_call: EutilsAPIRequest) -> List[Union[dict, str]]:
    response = execute_eutils_api_call(api_call)
    return [response]


def save_response(
    response_list: List[Union[dict, str]], file_path: str, question_uuid: str
) -> str:
    file_name = f"eutils_results_{question_uuid}.json"
    full_file_path = os.path.join(file_path, file_name)

    try:
        with open(full_file_path, "w") as file:
            json.dump(response_list, file, indent=4)
    except Exception as e:
        print(f"Error saving as JSON: {e}")
        if isinstance(response_list[0], bytes):
            file_name = f"eutils_results_{question_uuid}.bin"
        elif isinstance(response_list[0], str):
            file_name = f"eutils_results_{question_uuid}.txt"
        else:
            file_name = f"eutils_results_{question_uuid}.json"

        full_file_path = os.path.join(file_path, file_name)

        with open(
            full_file_path, "wb" if isinstance(response_list[0], bytes) else "w"
        ) as file:
            if isinstance(response_list[0], bytes):
                file.write(response_list[0])
            elif isinstance(response_list[0], str):
                file.write(response_list[0])
            else:
                json.dump(response_list, file)

    print(f"Results are saved in: {full_file_path}")

    return file_name
