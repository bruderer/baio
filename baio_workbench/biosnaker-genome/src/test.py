import os
import gffutils
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
import pandas as pd
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation

import gffutils
from Bio import SeqIO

class GeneInfoExtractor:
    def __init__(self, db_path, fasta_path):
        self.db_path = db_path
        self.fasta_path = fasta_path
        self.db = None
        self.fasta_dict = None
        self._gene = None  # Initialize the _gene attribute
        self.version = '0.1'
        
    def initialize(self):
        self.db = gffutils.FeatureDB(self.db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(self.fasta_path, "fasta"))
        
    @property
    def gene(self):
        return self._gene
    
    def get_gene_by_name(self, name):
        feature_list = []
        for feature in self.db.all_features():
            gene_name = feature.attributes.get('Name')
            if gene_name and name in gene_name:
                feature_list.append(feature)
        return feature_list if feature_list else None

    def get_gene_by_ref_gene(self, ref_gene_name):
        feature_list = []
        for feature in self.db.all_features():
            ref_gene = feature.attributes.get('ref-gene')
            if ref_gene and ref_gene_name in ref_gene:
                feature_list.append(feature)
        return feature_list if feature_list else None

    @gene.setter
    def gene(self, identifier):
        try:
            self._gene = self.db[identifier]
        except gffutils.FeatureNotFoundError:
            print(f"Failed to fetch gene by id: {identifier}")
            self._gene = self.get_gene_by_ref_gene(identifier)
            if self._gene is None:
                self._gene = self.get_gene_by_name(identifier)
            if self._gene is None:
                raise KeyError(f"No gene found with id, ref-gene, or name {identifier}")
            

         
class GeneAnalyzer:
    def __init__(self, db_path, fasta_path, gene,output_dir):
        self.db_path = db_path
        self.fasta_path = fasta_path
        self.db = gffutils.FeatureDB(self.db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(self.fasta_path, "fasta"))
        self.gene = gene  
        self.sequence = self.get_gene_sequence(self.gene)
        self.output_dir = output_dir
        self.coordinate_dict = None
        self.coordinate_df = None
        self.record = SeqRecord(self.sequence, id=self.gene.id, name=self.gene.attributes['Name'][0], dbxrefs=self.gene.attributes['ref-gene'], description=f"from genome: fill in!", annotations={"molecule_type": "DNA"})
        self.record_concatenated = None
        self.concatenated_cds_sequence = None
        
    def get_gene_sequence(self, gene):
        sequence = self.fasta_dict[gene.seqid].seq[gene.start - 1:gene.end]
        if gene.strand == '-':
            print('Gene on negative strand, reverse complementing sequence')
            sequence = sequence.reverse_complement()
        return sequence

    def get_cds_by_gene_gff_index(self, gene):
        return list(self.db.children(gene.id, featuretype='CDS'))
       
    def cds_coordinates_to_dict(self):

        #original gff coordinates
        ##################################
        cds_coordinates = {}
        cds_by_gene_gff_index = self.get_cds_by_gene_gff_index(self.gene)
        cds_coordinates['source'] = (self.gene.start, self.gene.end)        
        for i, cds in enumerate(cds_by_gene_gff_index):
            cds_coordinates[f'cds_{i}'] = (cds.start, cds.end)
        original_coordinates = {f'coordinate_type_gff': cds_coordinates}
        
        #python adjusted gff coordinates
        ##################################
        cds_coordinates_gff_python_adjusted={}
        #sequence indexes must be adjusted to python indexing
        cds_coordinates_gff_python_adjusted['source'] = (self.gene.start-1, self.gene.end-1)                
        for i, cds in enumerate(cds_by_gene_gff_index):
            #cds indexes must be adjusted to python indexing
            cds_coordinates_gff_python_adjusted[f'cds_{i}'] = (cds.start -1, cds.end - 1)
        python_adjusted_original_coordinates = {f'coordinate_type_gff_adjusted_python': cds_coordinates_gff_python_adjusted}
        
        #original gene adjusted coordinates
        ################################### 
        cds_coordinates_gff_gene_adjusted={}
        cds_coordinates_gff_gene_adjusted['source'] = (1, self.gene.end - self.gene.start + 1)
        for i, cds in enumerate(cds_by_gene_gff_index):
            cds_coordinates_gff_gene_adjusted[f'cds_{i}'] = (cds.start - self.gene.start + 1, cds.end - self.gene.start + 1)
        cds_coordinates_gff_gene_adjusted_ = {f'coordinate_type_gff_gene_adjusted': cds_coordinates_gff_gene_adjusted}        
        
        #python gene adjusted gff coordinates 
        ################################### 
        ###CHECK
        cds_coordinates_gene_adjusted_python={}
        cds_coordinates_gene_adjusted_python['source'] = (0, self.gene.end - self.gene.start)
        for i, cds in enumerate(cds_by_gene_gff_index):
            cds_coordinates_gene_adjusted_python[f'cds_{i}'] = (cds.start - self.gene.start, cds.end - self.gene.start)
        python_adjusted_original_coordinates_gene_adjusted = {f'coordinate_type_gene_adjusted_python': cds_coordinates_gene_adjusted_python}          
        
        
        #end result: list of all dictionaries
        ###################################
        sequence_coordinates = [original_coordinates, python_adjusted_original_coordinates, cds_coordinates_gff_gene_adjusted_, python_adjusted_original_coordinates_gene_adjusted]

        data_dict = {k: v for d in sequence_coordinates for k, v in d.items()}
        self.coordinate_dict = data_dict
        return data_dict
    

    def add_features_to_record(self, key):
        """
        Add features to a SeqRecord based on a specific key.

        Args:
            record (SeqRecord): The SeqRecord to add features to.
            coordinates_dict (dict): Dictionary containing the coordinates.
            key (str): The key to use to select the coordinates from the dictionary.

        Returns:
            None. The features are added directly to the SeqRecord.
        """
        self.get_gene_sequence(self.gene)
        
        locations = []
        for feature_key, (start, end) in self.coordinate_dict[key].items():
            # Create a new SeqFeature with the coordinates
            feature = SeqFeature(FeatureLocation(start-1, end), type=feature_key)

            # If the feature_key starts with 'cds_n', create a CDS feature
            if feature_key.startswith('cds_'):
                feature.type = 'CDS'
                feature.qualifiers['label'] = [feature_key]

            # Add the feature to the record
            self.record.features.append(feature)

            # Collect locations for a CompoundLocation
            locations.append(FeatureLocation(start, end))

    def concatenate_cds_sequences(self):
        """Concatenate sequence of CDS features in a SeqRecord. 
        Get the genebak coordinates (same as gff file, inclusif begin and end)"""
        # Initialize an empty Seq object to hold the concatenated sequence
        concatenated_sequence = None

        # Initialize a list to hold the new start and stop indices of each CDS
        new_indices = []

        # Get the 'coordinate_type_gene_adjusted_python' coordinates
        coordinates = self.coordinate_dict['coordinate_type_gene_adjusted_python']

        # Initialize a variable to hold the running total of the lengths of the CDS sequences
        total_length = 0

        # Iterate over the items in the coordinates
        for key, (start, end) in coordinates.items():
            # Check if the key starts with 'cds_'
            if key.startswith('cds_'):
                # Slice the sequence of the SeqRecord using the start and end positions
                cds_sequence = self.sequence[start:end+1]
                # Calculate the new start and stop indices of the CDS
                new_start = total_length
                new_stop = total_length + len(cds_sequence)
                # Add the new indices to the list
                new_indices.append((new_start, new_stop))
                # Update the running total of the lengths
                total_length += len(cds_sequence)
                # If this is the first CDS, initialize the concatenated sequence
                if concatenated_sequence is None:
                    concatenated_sequence = cds_sequence
                # Otherwise, concatenate the CDS sequence to the existing sequence
                else:
                    concatenated_sequence += cds_sequence

        # Create a new SeqRecord object with the concatenated sequence
        self.record_concatenated = SeqRecord(concatenated_sequence)
        self.record_concatenated.id = self.record.id
        self.record_concatenated.name = self.record.name
        self.record_concatenated.description = self.record.description
        self.record_concatenated.dbxrefs = self.record.dbxrefs
        self.record_concatenated.annotations['molecule_type'] = self.record.annotations['molecule_type']

        # Add SeqFeature objects to the new SeqRecord for each CDS
        for i, (start, end) in enumerate(new_indices):
            feature = SeqFeature(FeatureLocation(start, end), type='CDS', id=f'cds_{i}')
            self.record_concatenated.features.append(feature)

        return  self.record_concatenated
    
    def cds_coordinates_to_df_save_csv(self):
        """
        Convert CDS coordinates to a DataFrame and save it as a CSV file.

        This method creates a DataFrame from the dictionary,
        adds length columns to the DataFrame, calculates the total length, sorts the DataFrame,
        and saves it as a CSV file.

        Args:
            data_dict (dict): Dictionary containing the CDS coordinates.
            rec_conc (SeqRecord): SeqRecord object containing the sequence.

        Returns:
            df (DataFrame): DataFrame containing the CDS coordinates.
        """
        rec_conc = self.record_concatenated
        rec_conc_dict = {'concatenated_coordinates_snap_cds_concat_indexing': {'source': (1, len(rec_conc.seq))}}
        for feature in rec_conc.features:
            rec_conc_dict['concatenated_coordinates_snap_cds_concat_indexing'][feature.id] = (int(feature.location.start+1), int(feature.location.end))

        merged_dict = {**data_dict, **rec_conc_dict}

        df = pd.DataFrame(merged_dict)

        df = df.T

        for col in df.columns:
            if 'cds_' in col or col in rec_conc_dict:
                df[f'{col}_length'] = df[col].apply(lambda x: x[1] - x[0] + 1)

        df = df.sort_index()
        df = df.T
        df.to_csv(f'{self.output_dir}{self.gene.id}_coordinates.csv', index=True)

        self.coordinate_df = df
        return df
    
    def save_gene_record_tot_seq_genebank(self, seq_record_python_index_gene_ref):
        with open(f"{seq_record_python_index_gene_ref.id}_tot_seq.gb", "w") as output_handle:
            SeqIO.write(seq_record_python_index_gene_ref, output_handle, "genbank")



###test
current_dir = os.getcwd()

# # Define the paths to the GFF database and the FASTA file
db_path = os.path.join(current_dir, 'genome', 'gff.db')
fasta_path = os.path.join(current_dir, 'genome', 'primary_assembly_BY_260923_CI_02.fa')
output_dir = os.path.join(current_dir, 'output/')

# # Initialize the GeneAnalyzer with the paths to the GFF database and the FASTA file
# analyzer = GeneInfoExtractor(db_path, fasta_path)
# analyzer.initialize()

# # Define the identifier of the gene to analyze
# identifier = 'KY.Chr2.2230'

# # Get the gene from the GFF database
# analyzer.gene = identifier
# genes = analyzer.gene

# ###from here on we go to GeneAnalyzer
# ###
# ###
# gene = genes[0]
gene_analyzer = GeneAnalyzer(db_path, fasta_path, gene, output_dir)
gene_analyzer
# Get the sequence of the gene from the FASTA file

# Create a SeqRecord object for the gene
# Get the CDS features of the gene from the GFF database
gene_cds_gff_index = gene_analyzer.get_cds_by_gene_gff_index(gene)
data_dict = gene_analyzer.cds_coordinates_to_dict()
rec_conc = gene_analyzer.concatenate_cds_sequences()
gene_analyzer.add_features_to_record('coordinate_type_gff_gene_adjusted')
record = gene_analyzer.record
df = gene_analyzer.cds_coordinates_to_df_save_csv()
record = gene_analyzer.record

# Save rec_conc as a GenBank file
with open("rec_conc.gb", "w") as output_handle:
    SeqIO.write(rec_conc, output_handle, "genbank")

# Save record as a GenBank file
with open("record.gb", "w") as output_handle:
    SeqIO.write(record, output_handle, "genbank")



#     gene_analyzer.add_features_to_record(key)
#     with open(f"{record.id}_tot_seq.gb", "w") as output_handle:
#         SeqIO.write(record, output_handle, "genbank")
#     if key == 'coordinate_type_gff_gene_adjusted':
#         for feature in gene_analyzer.record.features:
#             feature.location = FeatureLocation(feature.location.start - 1, feature.location.end)

#     # if key == 'coordinate_type_gene_adjusted_python':

#     with open("output.gbk", "w") as output_handle:
#         SeqIO.write(gene_analyzer.record, output_handle, "genbank")
        
   
# save_gene_record_tot_seq_genebank(gene_analyzer, 'coordinate_type_gff_gene_adjusted')   
#     # New code to add length columns

# # def add_cds_features(record, data_dict, key):
    
# #     # Find the dictionary in data that has the key
# #     values = data_dict[key]
# #     for item in values:
# #         if 'cds_' in item:
# #             print(item)
# #             print(values[item])
# #             feature = SeqFeature(FeatureLocation(*values[item]), type='CDS', qualifiers={'coordinate_type': key})
# #             print(feature)
# #             record.features.append(feature)
# #     return record

# # values = data_dict[key]
# # for item in values:
# #     if 'cds_' in item:
# #         print(item)
# #         print(values[item])
        
        
# # key = 'coordinate_type_gene_adjusted_python'  # Replace with your specific keyword
# # record = add_cds_features(record, data_dict, key)


# # # Usage:
# # record = add_cds_features(record, data, 'coordinate_type_gff')
# # gene_analyzer.concatenated_cds_seq_from_seq_record_python_index_gene_ref(add_cds_features(record, data, 'coordinate_type_gene_adjusted_python'))

# # keyword = 'coordinate_type_gff'  # Replace with your specific keyword
# # values = [d[key] for d in data for key in d if key == keyword]


# # # Adjust the start and end positions of the CDS features
# # gene_cds_gff_index_list = gene_analyzer.adjust_cds_by_gene_gff_index(gene_cds_gff_index)

# # # Print the start and end positions of the CDS features
# # analyzer.print_out_record_list(gene_cds_gff_index_list, fasta_dict=analyzer.fasta_dict)

# # # Save the SeqRecord object as a GenBank file
# # seq_record_python_index_gene_ref = gene_cds_gff_index_list[2]
# # seq_record_python_index_gene_ref = gene_cds_gff_index_list[0]

# # analyzer.save_gene_record_tot_seq_genebank(seq_record_python_index_gene_ref)
# # analyzer.concatenated_cds_seq_from_seq_record_python_index_gene_ref(seq_record_python_index_gene_ref)

# # ####TEST
# # #test gene: depdc; ref-gene: KY.Chr2.2230
# # #1: extract sequence from fasta and see if it matches 
# #     #length
# #     #start
# #     #end
# #     #strand
    
# # #2: extract cds from gff and see if it matches
# # #3: extract cds from gff and adjust start and end positions to match fasta
# # #4: compare length of each cds from gff and adjusted cds from gff
# # #5:  compare start and end positions of each cds from gff and adjusted cds from gff
# # #6:  compare tot sequence from fasta and tot sequence from adjusted cds from gff
# # #7: compare concatenated sequence with tot sequence from fasta


# # #save the start end of gene and cds for gff, python tot genome adjusted index, python gene adjusted index in a panda dataframe

# def concatenate_cds_sequences(record, data_dict):
#     # Initialize an empty Seq object to hold the concatenated sequence
#     concatenated_sequence = None

#     # Initialize a list to hold the new start and stop indices of each CDS
#     new_indices = []

#     # Get the 'coordinate_type_gene_adjusted_python' coordinates
#     coordinates = data_dict['coordinate_type_gene_adjusted_python']

#     # Initialize a variable to hold the running total of the lengths of the CDS sequences
#     total_length = 0

#     # Iterate over the items in the coordinates
#     for key, (start, end) in coordinates.items():
#         # Check if the key starts with 'cds_'
#         if key.startswith('cds_'):
#             # Slice the sequence of the SeqRecord using the start and end positions
#             cds_sequence = record.seq[start:end+1]
#             # Calculate the new start and stop indices of the CDS
#             new_start = total_length+1
#             new_stop = total_length + len(cds_sequence)
#             # Add the new indices to the list
#             new_indices.append((new_start, new_stop))
#             # Update the running total of the lengths
#             total_length += len(cds_sequence)
#             # If this is the first CDS, initialize the concatenated sequence
#             if concatenated_sequence is None:
#                 concatenated_sequence = cds_sequence
#             # Otherwise, concatenate the CDS sequence to the existing sequence
#             else:
#                 concatenated_sequence += cds_sequence

#     return concatenated_sequence, new_indices

# # Use the function
# concatenated_sequence, new_indices = concatenate_cds_sequences(record, data_dict)
# print(concatenated_sequence)
# print(new_indices)

# # Use the function
# concatenated_sequence = concatenate_cds_sequences("./output.gbk")
# print(concatenated_sequence)



# concatenated_sequence_own_index = concatenate_cds_sequences(record, data_dict)
# print(concatenated_sequence_own_index)
# print(len(concatenated_sequence_own_index))

# # Use the function
# concatenated_sequence = concatenate_cds_sequences("output.gbk", data_dict)
# print(concatenated_sequence)

