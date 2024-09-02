import json
import os
import traceback

import requests

from baio.src.mytools.aniseed import ANISEED_multistep, ANISEED_query_generator
from baio.src.non_llm_tools import Utils

from . import AniseedJSONExtractor

# from langchain.tools import tool


def aniseed_tool(question: str, llm):
    """Takes in a question about any organisms on ANISEED and outputs a dataframe with
    requested information"""
    path_json = "./baio/data/output/aniseed/temp/aniseed_out_{counter}.json"
    path_save_csv = "./baio/data/output/aniseed/aniseed_out_{counter}.csv"

    print(f"\nTesting question: {question}")
    functions_to_call = ANISEED_multistep(question, llm)
    file_path = (
        "./baio/data/persistant_files/user_manuals/api_documentation/"
        "aniseed/aniseed_api_doc.txt"
    )

    with open(file_path, "r") as file:
        Aniseed_doc = file.read()
    print(f"One_or_more_steps: {functions_to_call.One_or_more_steps}")
    print(f"Function 1: {functions_to_call.functions_to_use_1}")
    print(f"Function 2: {functions_to_call.functions_to_use_2}")
    print(f"Function 3: {functions_to_call.functions_to_use_3}")
    functions_to_call = [
        functions_to_call.functions_to_use_1,
        functions_to_call.functions_to_use_2,
        functions_to_call.functions_to_use_3,
    ]
    api_calls = []
    functions_to_call = [func for func in functions_to_call if func is not None]
    for function in functions_to_call:
        print(f"\033[1;32;40mAniseed tool is treating: {function} \n")
        api_calls.append(ANISEED_query_generator(question, function, Aniseed_doc, llm))

    counter = 0
    aniseed_out_paths = []
    print(f"THE API CALLS ARE \n\n {api_calls}")
    print(len(api_calls))
    for api_call in api_calls:
        print(f"Processing API call {counter + 1} of {len(api_calls)}")
        print(f"API call: {api_call.full_url}")
        error_message = ""
        formatted_path_tempjson = path_json.format(counter=counter)
        formatted_path_save_csv = path_save_csv.format(counter=counter)
        response = requests.get(api_call.full_url)
        data = response.json()
        os.makedirs(os.path.dirname(formatted_path_tempjson), exist_ok=True)
        try:
            print(
                f"Path: formatted_path_tempjson = {formatted_path_tempjson}"
            )  # Check the path
            with open(formatted_path_tempjson, "w") as f:
                json.dump(data, f)
            print("Data saved successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

        prompt_2 = AniseedJSONExtractor(
            formatted_path_tempjson, formatted_path_save_csv
        ).get_prompt()
        print(prompt_2)
        print("python agent will be executed")
        aniseed_out_paths.append(formatted_path_save_csv)

        for attempt in range(5):
            try:
                print("In the process of formatting the JSON to csv")
                code = llm.invoke(prompt_2 + error_message)
                print(code.dict()["content"])
                Utils.execute_code_2(code.dict()["content"])
                break  # If the code executes successfully, break the loop
            except Exception as e:
                print(e)
                print("Attempt failed! Trying again.")
                error_message = (
                    f"\033[1;31;40mPrevious attempt:"
                    f"{code.dict()['content']}; Error: {traceback.format_exc()} Change "
                    "the code so that you solve the error\033[0;37;40m"
                )  # Append the error message to the prompt
                print(error_message)
        else:
            print("All attempts failed!")

        counter += 1  # Increment the counter at the end of each iteration

    return aniseed_out_paths
