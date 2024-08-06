import json
import os
import traceback

import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.vectorstores import FAISS

from baio.src.llm import LLM
from baio.src.mytools.aniseed import ANISEED_multistep, ANISEED_query_generator

embedding = OpenAIEmbeddings()


ANISEED_db = FAISS.load_local(
    "./baio/data/persistant_files/vectorstores/aniseed", embedding
)

llm = LLM.get_instance()


@tool
def aniseed_tool(question: str):
    """Takes in a question about any organisms on ANISEED and outputs a dataframe with
    requested information"""
    path_tempjson = "./baio/data/output/aniseed/temp/aniseed_temp.json"
    path_json = "./baio/data/output/aniseed/temp/aniseed_out_{counter}.json"
    path_save_csv = "./baio/data/output/aniseed/aniseed_out_{counter}.csv"
    multistep, top_3_retrieved_docs = ANISEED_multistep(question, llm, ANISEED_db)
    print(
        "[1;32;40m]Functions to be called: "
        f"{multistep.functions_to_use_1}\n{multistep.functions_to_use_2}\n"
        f"{multistep.functions_to_use_3}\n"
    )

    api_calls = []
    for function in (
        multistep.functions_to_use_1,
        multistep.functions_to_use_2,
        multistep.functions_to_use_3,
    ):
        if function is not None:
            print(f"\033[1;32;40mAniseed tool is treating: {function} \n")

            api_calls.append(
                ANISEED_query_generator(question, function, top_3_retrieved_docs, llm)
            )
    # print(
    #     f"Now we have the api calls that have to be executed to obtain the data to "
    #     "answer the users question:\n"
    #     + "\n".join([api_call.full_url for api_call in api_calls])

    counter = 0
    aniseed_out_paths = []
    for api_call in api_calls:
        error_message = ""
        formatted_path_tempjson = path_json.format(counter=counter)
        formatted_path_save_csv = path_save_csv.format(counter=counter)
        response = requests.get(api_call.full_url)
        data = response.json()
        # print(data)

        os.makedirs(os.path.dirname(formatted_path_tempjson), exist_ok=True)
        try:
            print(f"Path: {formatted_path_tempjson}")  # Check the path
            with open(path_tempjson, "w") as f:
                json.dump(data, f)
            with open(formatted_path_tempjson, "w") as f:
                json.dump(data, f)
            print("Data saved successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

        prompt_2 = AniseedJSONExtractor(
            path_tempjson, formatted_path_save_csv
        ).get_prompt()
        print("python agent will be executed")
        aniseed_out_paths.append(formatted_path_save_csv)
        for attempt in range(5):
            try:
                print("In the process of formating the JSON to csv")
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
        counter += 1
        return aniseed_out_paths
