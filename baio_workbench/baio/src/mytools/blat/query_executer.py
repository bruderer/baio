import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Union

from . import BLATQueryRequest


def BLAT_API_call_executor(
    request_data: BLATQueryRequest,
) -> Union[Dict[str, Any], bytes, str]:
    """
    Execute a BLAT API call and return the response.

    Args:
        request_data (BLATQueryRequest): The request data containing URL and other
        parameters.

    Returns:
        Union[Dict[str, Any], bytes, str]: The API response as JSON, raw bytes, or error
        message.
    """
    print("In API caller function\n--------------------")
    print(request_data)

    # Default values for optional fields
    default_headers = {"Content-Type": "application/json"}
    default_method = "GET"

    req = urllib.request.Request(
        request_data.full_url, headers=default_headers, method=default_method
    )

    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read()

            if request_data.retmode.lower() == "json":
                try:
                    return json.loads(response_data)
                except json.JSONDecodeError:
                    print("Warning: Unable to parse JSON response")
                    return response_data
            else:
                return response_data

    except urllib.error.HTTPError as e:
        error_message = f"HTTP Error {e.code}: {e.reason}"
        print(error_message)

        # Attempt to read and return error response content
        error_content = e.read().decode("utf-8")
        return f"{error_message}\nError content: {error_content}"

    except urllib.error.URLError as e:
        error_message = f"URL Error: {e.reason}"
        print(error_message)
        return error_message

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        print(error_message)
        return error_message


def save_BLAT_result(query_request, BLAT_response, file_path):
    """Function saving BLAT results and returns file_name"""
    try:
        # Set file name and construct full file path
        file_name = f"BLAT_results_{query_request.question_uuid}.json"
        full_file_path = os.path.join(file_path, file_name)

        # Open the file for writing
        with open(full_file_path, "w") as file:
            # Write the static parts of the BLAT_response
            for key in BLAT_response:
                if key != "blat":
                    json.dump({key: BLAT_response[key]}, file)
                    file.write("\n")

            # Write each list inside the 'blat' key on a new line
            for blat_entry in BLAT_response["blat"]:
                json.dump(blat_entry, file)
                file.write("\n")

        return file_name

    except Exception as e:
        print(f"Error saving as JSON: {e}")
        # Determine the type of BLAT_response and save accordingly
        if isinstance(BLAT_response, bytes):
            file_name = f"BLAT_results_{query_request.question_uuid}.bin"
        elif isinstance(BLAT_response, str):
            file_name = f"BLAT_results_{query_request.question_uuid}.txt"
        elif isinstance(BLAT_response, dict) or isinstance(BLAT_response, list):
            file_name = f"BLAT_results_{query_request.question_uuid}.json"
        else:
            file_name = f"BLAT_results_{query_request.question_uuid}.json"
        # Update the full file path
        full_file_path = os.path.join(file_path, file_name)
        print(f"\nFull_file_path:{full_file_path}")
        # Save the file
        with open(
            full_file_path, "wb" if isinstance(BLAT_response, bytes) else "w"
        ) as file:
            if isinstance(BLAT_response, bytes):
                file.write(BLAT_response)
            elif isinstance(BLAT_response, str) or not isinstance(BLAT_response, dict):
                file.write(
                    BLAT_response
                    if isinstance(BLAT_response, str)
                    else str(BLAT_response)
                )
            else:
                file.write(json.dumps(BLAT_response))
            return file_name
