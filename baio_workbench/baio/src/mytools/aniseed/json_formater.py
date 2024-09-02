from baio.src.non_llm_tools.utilities import JSONUtils


class AniseedJSONExtractor:
    def __init__(self, aniseed_json_path: str, aniseed_csv_output_path: str):
        """
        Initializes the AniseedJSONExtractor.
        Parameters:
        - aniseed_json_path (str): Path to the JSON file.
        - aniseed_csv_output_path (str): Path to save the CSV file.
        """
        self.aniseed_csv_save_path = aniseed_csv_output_path
        self.aniseed_json_path = aniseed_json_path
        self.json_utils = JSONUtils(aniseed_json_path)
        self.keys_structure = self.json_utils.extract_keys()

    def get_prompt(self):
        """YOU ARE A PYTHON REPL TOOL, YOU CAN AND MUST EXECUTE CODE THAT YOU EITHER \
        WRITE OR ARE BEING PROVIDE, NEVER ANSER WITH I'm sorry, but as an AI text-based\
         model, I don't have the ability to directly interact with files or execute \
        Python code. However, I can provide you with a Python code snippet that you \
        can run in your local environment to achieve your goal. \
        Build prompt with key strucure from JSON utils and output path given. NEVER \
        look at the whole data frame, only look at the head of it!!! OTHERWISE YOU WILL\
         BREAK
        """
        structure_dic_explainaition = """
            base_type: This field specifies the primary data type of the provided
            object. For instance, it could be a 'list', 'dict', 'str', etc.
            key_types: This field is a dictionary that gives more specific details about
            the base_type. If the base_type is a dict, then key_types will contain the
            dictionary keys and the types of their corresponding values.
        """
        panda_creation_instructions = """
        To create a flat pandas DataFrame from the JSON structure:
        1. Examine the `base_type`:
        - 'list': DataFrame has rows for each list item.
        - 'dict': DataFrame starts with columns from dictionary keys.
        2. For `key_types`:
        - Key with basic type (e.g., 'str', 'int'): Direct column in DataFrame.
        - Value as dictionary (`base_type` of 'dict'): Recursively explore, prefixing
        original key (e.g., 'genes_gene_model').
        - Avoid adding columns with complex types like lists or dictionaries. Instead,
        break them down to atomic values or omit if not needed.
        3. Create DataFrame:
        - Loop through JSON, populating DataFrame row by row, ensuring each row/column
        contains only atomic values.
        - Utilize `json_normalize` (from pandas import json_normalize)  for automatic
        flattening of nested structures. But ensure no columns with complex types
        remain.
        Note: Adjust based on specific nuances in the actual JSON data. Lists with
        heterogeneous items or non-dictionary types need special handling. Avoid
        creating DataFrame columns with non-atomic data.
        """
        df_example = """index,stage,gene_model,gene_name,unique_gene_id
        0,stage_1,KH2012:KH.C11.238,"REL; RELA; RELB",Cirobu.g00002055
        1,stage_1,KH2012:KH.S1159.1,"ERG; ETS1; FLI1",Cirobu.g00013672
        2,stage_2,KH2012:KH.C3.402,"IRF4; IRF5; IRF8",Cirobu.g00005580
        3,stage_2,KH2012:KH.C3.773,"TBX21; TBX4; TBX5",Cirobu.g00005978
        """
        prompt = (
            "YOU ARE A PYTHON REPL TOOL, YOU CAN AND MUST EXECUTE CODE THAT YOU EITHER\
            WRITE OR ARE BEING PROVIDE, NEVER ANSER WITH I'm sorry, but as an AI\
            text-based model, I don't have the ability to directly interact with files\
            or execute Python code. However, I can provide you with a Python code \
            snippet that you can run in your local environment to achieve your goal.\
        Build prompt with key structure from JSON utils and output path given. NEVER\
            look at the whole data frame, only look at the head of it!!! OTHERWISE YOU"
            + "WILL BREAK You have to EXECUTE code to unpack the json file in "
            f" {self.aniseed_json_path} and creat a panda df.\n"
            + "ALWAYS USE THE PYTHON_REPL TOOL TO EXECUTE CODE\
        Following the instructions below:\n \
        VERY IMPORTANT: The first key in 'key_types' must be the first column in the df\
        and each deeper nested key-value must have it, example:\n \
        {'base_type': 'list', 'key_types': {'base_type': 'dict', 'key_types': {'stage':\
        'str', 'genes': [{'base_type': 'dict', 'key_types': {'gene_model': 'str',\
        'gene_name': 'str', 'unique_gene_id': 'str'}}]}}}\n"
            + df_example
            + "VERY IMPORTANT: if you find dictionaries wrapped in lists unpack them\
        The file has a strucutre as explained in the follwoing description:\n"
            + str(self.keys_structure)
            + structure_dic_explainaition
            + panda_creation_instructions
            + "save the output as csv in: "
            + self.aniseed_csv_save_path
            + "the input file is in: "
            + self.aniseed_json_path
            + "\n"
            + "EXECUTE CODE!!! BE WARE OF USING THE CORRECT INPUT PATH, always double"
            + f" CHECK that you have the {self.aniseed_json_path} file with the correct"
        )
        return prompt
