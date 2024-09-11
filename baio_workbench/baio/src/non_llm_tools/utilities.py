import ast
import json
import os
import re
import threading
from typing import Any, Optional

import pandas as pd

# Lock for synchronizing file access
file_lock = threading.Lock()


class Utils:
    """
    A utility class to enhance agents

    """

    @staticmethod
    def extract_python_code(text: str) -> str:
        start_marker = "```python"
        end_marker = "```"
        code_start = text.find(start_marker) + len(start_marker)
        code_end = text.find(end_marker, code_start)
        if code_start == -1 or code_end == -1:
            return "No code found!"
        extracted_code = text[code_start:code_end].strip()
        return extracted_code

    @staticmethod
    def execute_code_2(input_):
        code = Utils.extract_python_code(input_)
        print(f"\033[1;35;40m {code} \033[0;37;40m")
        exec(code)

    @staticmethod
    def execute_code(result: dict):
        code = Utils.extract_python_code(result["answer"])
        print(code)
        exec(code)

    @staticmethod
    def flatten_aniseed_gene_list(
        input_file_path: str, input_file_gene_name_column: str
    ) -> list:
        """
        Flatten and extract gene names from an aniseed returned and parsed CSV file.

        Parameters:
        - input_file_path (str): Path to the CSV file containing gene names to be
        annotated.
        - input_file_gene_name_column (str): Column name in the CSV that contains the
        gene names.

        Returns:
        - list: A flattened list of gene names.
        """
        gene_list = list(pd.read_csv(input_file_path)[str(input_file_gene_name_column)])
        gene_list = [gene for sublist in gene_list for gene in sublist.split("; ")]
        return gene_list

    @staticmethod
    def parse_refseq_id(go_dataframe: pd.DataFrame) -> pd.DataFrame:
        def unpack_refseq(refseq_str):
            try:
                refseq_dict = ast.literal_eval(refseq_str)
                if isinstance(refseq_dict, dict):
                    return pd.Series(refseq_dict)
                else:
                    return pd.Series()
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Error parsing RefSeq string: {e}")
                return pd.Series()

        print(go_dataframe.head(5))
        refseq_df = go_dataframe["refseq_id"].apply(unpack_refseq)
        if "translation" in refseq_df.columns:
            refseq_df = refseq_df.drop(columns="translation")
        else:
            print(f"not found: {refseq_df}")
            return go_dataframe

        refseq_df = refseq_df.add_suffix("_refseq_id")
        refseq_df["genomic_refseq_id"] = refseq_df["genomic_refseq_id"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        refseq_df["protein_refseq_id"] = refseq_df["protein_refseq_id"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        refseq_df["rna_refseq_id"] = refseq_df["rna_refseq_id"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

        merged_df = pd.merge(go_dataframe, refseq_df, left_index=True, right_index=True)
        return merged_df


class JSONUtils:
    """
    A utility class for extracting key structures from a JSON file.
    Attributes:
    - path (str): Path to the JSON file.
    - data: Placeholder for loaded data. Initialized to None.
    Methods:
    - extract_keys_from_obj: Recursively extract keys and their types from an object
    (dictionary or list).
    - extract_keys: Load JSON from a file and extract its key structure.
    """

    def __init__(self, path: str):
        """
        Initializes the JSONUtils object with a path to a JSON file.
        Parameters:
        - path (str): Path to the JSON file.
        """
        self.path = path
        self.data = None
        print(f"Path: {self.path}")

    def extract_keys_from_obj(self, obj):
        """
        Recursively extract keys and their types from an object, which can be a
        dictionary or list.
        The goal is to obtain the main structure of any JSON content.
        Parameters:
        - obj (dict/list): A dictionary or list object from JSON content.
        Returns:
        - dict: Dictionary containing base type and key types and their names.
        """
        keys_dict = {}
        base_type = type(obj).__name__
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    keys_dict[key] = self.extract_keys_from_obj(value)
                elif isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    unique_sub_keys = {}
                    for item in value:
                        for k, v in self.extract_keys_from_obj(item).items():
                            unique_sub_keys[k] = v
                    keys_dict[key] = [unique_sub_keys]
                else:
                    keys_dict[key] = type(value).__name__
        elif isinstance(obj, list) and obj:
            if all(
                isinstance(item, dict) for item in obj
            ):  # Assuming homogeneous list items
                for item in obj:
                    for k, v in self.extract_keys_from_obj(item).items():
                        keys_dict[k] = v
        return {"base_type": base_type, "key_types": keys_dict}

    def extract_keys(self):
        """
        Loads a JSON file, given the path provided during object instantiation, and
        extracts its key structure.
        Returns:
        - dict: Dictionary containing the base type and key types of the JSON content.
        """
        print(f"Path: {self.path}")
        with open(self.path, "r") as file:
            obj = json.load(file)
        return self.extract_keys_from_obj(obj)


def log_question_uuid_json(
    question_uuid: str,
    question: str,
    file_name: str,
    file_path: str,
    log_file_path: str,
    full_url: str,
    answer: Optional[Any] = None,
    tool: Optional[str] = None,
):
    """
    Logs information about a question into a JSON file.

    This function takes details about a query object and appends them to a log file in
    JSON format. It is designed to track questions and associated metadata for
    record-keeping or analysis purposes.

    Args:
        question_uuid (str): A unique identifier for the question, typically in UUID
        format.
        question (str): The text of the question.
        file_name (str): The name of the file where the question is originally stored or
        referenced.
        file_path (str): The path to the file where the question is stored, relative or
        absolute.
        log_file_path (str): The path to the JSON log file where the question details
        will be appended.
        full_url (str): The full URL, if any, associated with the question or its
        source.

    Returns:
        None: This function does not return any value. It performs a logging action.

    Example:
        log_question_uuid_json("123e4567-e89b-12d3-a456-426614174000", "What is the
        capital of France?",
            "questions.txt", "/path/to/questions.txt", "/path/to/log.json",
            "https://example.com/question/123e4567-e89b-12d3-a456-426614174000")

    Note:
        The function assumes that the log file specified in `log_file_path` exists and
        is in a valid JSON format.
        It appends the question data as a new object in the JSON array.
    """
    # Function implementation goes here

    directory = os.path.dirname(log_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Initialize or load existing data
    data = []

    # Try reading existing data, handle empty or invalid JSON
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        try:
            with file_lock:
                with open(log_file_path, "r") as file:
                    data = json.load(file)
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON in {log_file_path}. Starting a new"
                "log."
            )

    # Construct the full file path
    full_file_path = os.path.join(file_path, file_name)

    # Add new entry with UUID, question, file name, and file path
    log_entry = {
        "uuid": question_uuid,
        "question": question,
        "file_name": file_name,
        "file_path": full_file_path,
        "API_info": full_url,
        "tool": tool,
    }
    if answer is not None:
        log_entry["answer"] = answer
    data.append(log_entry)

    # Write updated data
    with file_lock:
        with open(log_file_path, "w") as file:
            json.dump(data, file, indent=4)


# new functions


def extract_content_between_backticks(text):
    # Regular expression pattern for content within triple backticks
    pattern = r"```(.*?)```"

    # Search for matches
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the first match or None if no match is found
    return matches[0] if matches else None
