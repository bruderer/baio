import pandas as pd
import concurrent.futures
import mygene
from baio.src.non_llm_tools.utilities import Utils


def process_go_category(go_category):
    # Process a single GO category to extract terms and their IDs
    if isinstance(go_category, list) and all(isinstance(item, dict) for item in go_category):
        terms = [f"{item.get('qualifier', '')} {item.get('term', '')}".strip() for item in go_category]
        go_ids = [item.get('id', '') for item in go_category]
        return terms, go_ids
    else:
        return None, None

    
def unpack_go_terms(df):
    # Function to unpack GO terms and GO IDs
    df['biological_process_terms'], df['biological_process_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('BP')) if isinstance(x, dict) else (None, None)))
    df['cellular_component_terms'], df['cellular_component_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('CC')) if isinstance(x, dict) else (None, None)))
    df['molecular_function_terms'], df['molecular_function_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('MF')) if isinstance(x, dict) else (None, None)))

    # Drop the original 'go' column
    df.drop('go', axis=1, inplace=True)
    return df


def unpack_refseq(df):
    # Unpack RefSeq data
    df['refseq_genomic'] = df['refseq'].apply(lambda x: x.get('genomic', None) if isinstance(x, dict) else None)
    df['refseq_protein'] = df['refseq'].apply(lambda x: x.get('protein', None) if isinstance(x, dict) else None)
    df['refseq_rna'] = df['refseq'].apply(lambda x: x.get('rna', None) if isinstance(x, dict) else None)

    # Drop the original 'refseq' column
    df.drop('refseq', axis=1, inplace=True)
    return df


def unpack_ensembl(df):
    # Unpack Ensembl data
    df['ensembl_gene'] = df['ensembl'].apply(lambda x: x.get('gene', '') if isinstance(x, dict) else '')

    # Drop the original 'ensembl' column
    df.drop('ensembl', axis=1, inplace=True)
    return df



def unpack_uniprot(df):
    # Unpack UniProt data
    df['uniprot_trEMBL'] = df['uniprot'].apply(lambda x: x.get('TrEMBL', None) if isinstance(x, dict) else None)
    df['uniprot_swissprot'] = df['uniprot'].apply(lambda x: x.get('Swiss-Prot', None) if isinstance(x, dict) else None)

    # Drop the original 'uniprot' column
    df.drop('uniprot', axis=1, inplace=True)
    return df


def get_gene_info_df(gene_name):
    mg = mygene.MyGeneInfo()
    response = mg.query(gene_name, fields='symbol,taxid,entrezgene,ensembl.gene,uniprot,refseq,go')

    # Directly create a DataFrame from the response
    df = pd.DataFrame(response['hits'])

    df = unpack_uniprot(df)
    df = unpack_go_terms(df)
    df = unpack_refseq(df)
    df = unpack_ensembl(df)
    # Add a new column with the gene name for all rows
    gene_query_col = [gene_name] * len(df)
    df.insert(0, 'query_name', gene_query_col)

    return df

def process_gene(gene):
    print(f"Processing gene: {gene}")
    try:
        gene_df = get_gene_info_df(gene)
        gene_df['gene_name'] = gene  # Adding a gene name column
        return gene_df
    except Exception as e:
        print(f"Could not process gene {gene}: {e}")
        return None
    
def process_genes(gene_list):
    gene_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for gene_df in executor.map(process_gene, gene_list):
            if gene_df is not None:
                gene_dfs.append(gene_df)

    # Concatenating all gene dataframes
    concatenated_df = pd.concat(gene_dfs, ignore_index=True)
    return concatenated_df


def identify_gene_column(df: pd.DataFrame, gene_list: list):
    """Identify the column containing gene names based on a reference list."""
    
    def matches_in_column(column):
        """Return the number of matches in the given column."""
        return df[column].astype(str).str.contains('|'.join(gene_list)).sum()

    # First, always check the 'gene_name' column if it exists in the dataframe
    if 'gene_name' in df.columns:
        required_matches = max(1, min(5, max(len(gene_list)//50, df['gene_name'].shape[0])))
        if matches_in_column('gene_name') >= required_matches:
            return 'gene_name'
    
    for gene_matching_col in df.columns:
        # Skip the 'genemodel' column
        if gene_matching_col == 'genemodel' or gene_matching_col == 'gene_name':
            continue
        
        required_matches = max(1, min(5, max(len(gene_list)//50, df[gene_matching_col].shape[0])))
        # Check if the column contains the required number of reference gene names
        if matches_in_column(gene_matching_col) >= required_matches:
            return gene_matching_col
    return None

def concatenate_dataframes(input_file_path: str, concat_gene_df, gene_list):
    # Step 1: Read in the input_file_df from the provided path

    input_file_df = pd.read_csv(input_file_path)

    # Step 2: Identify the gene column in input_file_df
    gene_col = identify_gene_column(input_file_df, gene_list)
    if not gene_col:
        raise ValueError("No gene column identified in the input dataframe.")
    
    # Step 3: Split and explode the identified gene column directly
    input_file_df = input_file_df.assign(**{gene_col: input_file_df[gene_col].str.split('; ')}).explode(gene_col)
    
    # Step 4: Merge the two dataframes based on the gene names
    merged_df = pd.merge(input_file_df, 
                        concat_gene_df, 
                        left_on=gene_col, 
                        right_on='query_name', 
                        how='left').drop(columns='query_name')
    
    return merged_df


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)



def go_file_tool(input_file_path: str, input_file_gene_name_column: str) -> pd.DataFrame:
    """Used when the input is a file and not a human written query.
    Tool to find gene ontologies (using mygene), outputs data frame with GO & gene id annotated gene names

    Parameters:
    input_file_path (str): A string which is a path to the csv file containg gene names to be annotated
    input_file_gene_name_column (str): a string which is the column name containing the 

    Returns:
    final_dataframe (dataframe): A df contianing annotated genes with GO & IDs from mygene concatenated with the input file.
    """
    
    gene_list = list(set(Utils.flatten_aniseed_gene_list(input_file_path, input_file_gene_name_column)))
    print(gene_list)
    #we extract all the go terms and ids for all genes in this list
    final_go_df = process_genes(gene_list)
    print(final_go_df)
    final_go_df = concatenate_dataframes(input_file_path, final_go_df, gene_list)
    final_go_df.to_csv(input_file_path[:-4]+'_GO_annotated.csv', index=False)
    file_name = input_file_path[:-4]+'_GO_annotated.csv'
    return final_go_df, file_name


