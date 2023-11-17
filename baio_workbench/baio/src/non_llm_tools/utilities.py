import pandas as pd 
import json
import ast

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
    def execute_code(result: dict):
        code = Utils.extract_python_code(result['answer'])
        print(code)
        exec(code)
    
    @staticmethod
    def flatten_aniseed_gene_list(input_file_path: str, input_file_gene_name_column:str) -> list:
        """
        Flatten and extract gene names from an aniseed returned and parsed CSV file.

        Parameters:
        - input_file_path (str): Path to the CSV file containing gene names to be annotated.
        - input_file_gene_name_column (str): Column name in the CSV that contains the gene names.

        Returns:
        - list: A flattened list of gene names.
        """
        gene_list = list(pd.read_csv(input_file_path)[str(input_file_gene_name_column)])
        gene_list = [gene for sublist in gene_list for gene in sublist.split('; ')]
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
            except:
                return pd.Series()
        print(go_dataframe.head(5))
        refseq_df = go_dataframe['refseq_id'].apply(unpack_refseq)
        if 'translation' in refseq_df.columns:
            refseq_df = refseq_df.drop(columns='translation')
        else: 
            print(f'not found: {refseq_df}')
            return go_dataframe

        refseq_df = refseq_df.add_suffix('_refseq_id')
        refseq_df['genomic_refseq_id'] = refseq_df['genomic_refseq_id'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        refseq_df['protein_refseq_id'] = refseq_df['protein_refseq_id'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        refseq_df['rna_refseq_id'] = refseq_df['rna_refseq_id'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

        merged_df = pd.merge(go_dataframe, refseq_df, left_index=True, right_index=True)
        return merged_df

class JSONUtils:
    """
    A utility class for extracting key structures from a JSON file.
    
    Attributes:
    - path (str): Path to the JSON file.
    - data: Placeholder for loaded data. Initialized to None.
    
    Methods:
    - extract_keys_from_obj: Recursively extract keys and their types from an object (dictionary or list).
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

    def extract_keys_from_obj(self, obj):
        """
        Recursively extract keys and their types from an object, which can be a dictionary or list.
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
                elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    unique_sub_keys = {}
                    for item in value:
                        for k, v in self.extract_keys_from_obj(item).items():
                            unique_sub_keys[k] = v
                    keys_dict[key] = [unique_sub_keys]
                else:
                    keys_dict[key] = type(value).__name__
        elif isinstance(obj, list) and obj:
            if all(isinstance(item, dict) for item in obj):  # Assuming homogeneous list items
                for item in obj:
                    for k, v in self.extract_keys_from_obj(item).items():
                        keys_dict[k] = v
                        
        return {
            'base_type': base_type,
            'key_types': keys_dict
        }

    def extract_keys(self):
        """
        Loads a JSON file, given the path provided during object instantiation, and extracts its key structure.
        
        Returns:
        - dict: Dictionary containing the base type and key types of the JSON content.
        """
        with open(self.path, 'r') as file:
            obj = json.load(file)
        return self.extract_keys_from_obj(obj)

