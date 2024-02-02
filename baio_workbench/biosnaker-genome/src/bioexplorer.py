from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

import gffutils
import os 
from gffutils.exceptions import FeatureNotFoundError


class GenomeExplorer:
    def __init__(self, db_path, fasta_path, output_dir):
        """
        Initialize the GenomeExplorer class.

        Args:
            db_path (str): Path to the GFF database file.
            fasta_path (str): Path to the FASTA file.
            output_dir (str): Path to the output directory.
        """
        self.db = gffutils.FeatureDB(db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        self.output_dir = output_dir

    def get_gene_by_name(self, name):
        """
        Obtain the gene record based on name. If more than one gene is found, return all of them.

        Args:
            name (str): Name of the gene.

        Returns:
           list of gffutils.Feature: The gene records.
        """
        feature_list = []
        for feature in self.db.all_features():
            gene_name = feature.attributes.get('Name')
            if gene_name and name in gene_name:
                feature_list.append(feature)
        return feature_list if feature_list else None
    
    def get_gene_by_ref_gene(self, ref_gene_name):
        """
        Obtain the gene records based on ref-gene name. If more than one gene is found, return all of them.

        Args:
            ref_gene_name (str): Ref-gene name of the gene.

        Returns:
            list of gffutils.Feature: The gene records.
        """
        feature_list = []
        for feature in self.db.all_features():
            ref_gene = feature.attributes.get('ref-gene')
            if ref_gene and ref_gene_name in ref_gene:
                feature_list.append(feature)
        return feature_list if feature_list else None
    
    def get_gene_info(self, identifier):
        """
        Fetch gene information from the database based on gene_id, ref-gene, or name.

        Args:
            identifier (str): Identifier of the gene.

        Returns:
            gffutils.Feature: The gene record.

        Raises:
            KeyError: If no gene is found with the given identifier.
        """
        try:
            gene = self.db[identifier]
        except FeatureNotFoundError:
            print(f"Failed to fetch gene by id: {identifier}")
            gene = self.get_gene_by_ref_gene(identifier)
            if gene is None:
                gene = self.get_gene_by_name(identifier)
            if gene is None:
                raise KeyError(f"No gene found with id, ref-gene, or name {identifier}")
        return gene

    def get_gene_sequence(self, gene):
        """
        Fetch the sequence of a gene. If the gene is on the negative strand, the sequence will be reverse complemented.

        Args:
            gene_id (str): Identifier of the gene.

        Returns:
            Bio.Seq.Seq: The sequence of the gene.
        """
        sequence = self.fasta_dict[gene.seqid].seq[gene.start - 1:gene.end]
        if gene.strand == '-':
            print('Gene on negative strand, reverse complementing sequence')
            sequence = sequence.reverse_complement()
            # Adjust the coordinates of the features
            for feature in self.db.children(gene.id):
                feature.start, feature.end = len(sequence) - feature.end, len(sequence) - feature.start
        return sequence

    def get_gene_sequence(self, gene):
        """
        Fetch the sequence of a gene. If the gene is on the negative strand, the sequence will be reverse complemented.

        Args:
            gene_id (str): Identifier of the gene.

        Returns:
            Bio.Seq.Seq: The sequence of the gene.
        """
        sequence = self.fasta_dict[gene.seqid].seq[gene.start - 1:gene.end]
        if gene.strand == '-':
            print('Gene on negative strand, reverse complementing sequence')
            sequence = sequence.reverse_complement()
            # Adjust the coordinates of the features
            for feature in self.db.children(gene.id):
                feature.start, feature.end = len(sequence) - feature.end, len(sequence) - feature.start
        return sequence
        
    def build_record(self, gene_id):
        """
        Build a SeqRecord object for a gene.

        Args:
            gene_id (str): Identifier of the gene.

        Returns:
            Bio.SeqRecord.SeqRecord: The SeqRecord object representing the gene.
        """
        gene = self.get_gene_info(gene_id)
        sequence = self.get_gene_sequence(gene_id)
        record = SeqRecord(sequence, id=gene_id, description="", annotations={"molecule_type": "DNA"})
        gene_feature = SeqFeature(FeatureLocation(gene.start - 1, gene.end), type="gene", qualifiers={"gene_id": gene_id})
        record.features.append(gene_feature)
        cds_records = list(self.db.children(gene.id, featuretype='CDS'))
        for cds in cds_records:
            cds_feature = SeqFeature(FeatureLocation(cds.start - 1, cds.end), type="CDS")
            record.features.append(cds_feature)
        return record
    

    def gene_bank_file_creator_tot_sequence(self, identifier, raw):
        
        ###TO DO: moelcular_type hardcoded change!!!
        """
        Create a GenBank file for a gene containing CDS features.

        Args:
            identifier (str): Identifier of the gene.
        """
        generated_file_paths = []
        genes = self.get_gene_info(identifier)
        print(f'Found {len(genes)} genes matching the identifier {identifier}')
        for gene in genes:
            sequence = self.get_gene_sequence(gene)  # Modified line
            # Rest of the code
            record = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
            gene_feature = SeqFeature(FeatureLocation(0, gene.end - gene.start), type="gene", qualifiers={"gene_id": gene.id})
            record.features.append(gene_feature)
            cds_records = list(self.db.children(gene.id, featuretype='CDS'))
            if raw == False:
                print('Adjusting CDS start and end locations')
                for cds in cds_records:
                    # keep the -1 and +1 for clarification of the indexing adjustment even though they cancel out
                    cds_start = cds.start - gene.start + 1                
                    cds_end = cds.end - gene.start +1
                    cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
                    record.features.append(cds_feature)
                output_path = os.path.join(self.output_dir, f'{gene.id}.gb')
                generated_file_paths.append(output_path)
                SeqIO.write([record], output_path, "genbank")
                print(f'\nFile: {gene.id}.gb\nSaved in: {output_path}\n')
                            
            # if raw == True:
            #     print('Saving raw sequence start and stop locations')
            #     for cds in cds_records:
            #         print(cds)
            #         cds_feature = SeqFeature(FeatureLocation(cds.start, cds.end), type="CDS")
            #         record.features.append(cds_feature)
            #     output_path = os.path.join(self.output_dir, f'{gene.id}_raw.gb')
            #     generated_file_paths.append(output_path)
            #     SeqIO.write([record], output_path, "genbank")
            #     print(f'{gene.id}')
            #     print(f'\nFile: {gene.id}.gb\nSaved in: {output_path}\n')         
            
        return generated_file_paths


            
    def extract_gene_info(genbank_file_path):
        with open(genbank_file_path, 'r') as file:
            record = SeqIO.read(file, "genbank")
            for feature in record.features:
                if feature.type == "gene":
                    gene_id = feature.qualifiers.get('gene_id', [''])[0]
                    print(f"Gene ID: {gene_id}")
            molecular_type = record.annotations.get('molecule_type', '')
            print(f"Molecule type: {molecular_type}")
        return gene_id, molecular_type
           
    def extract_cds_sequences(self, filename): 
        cds_sequences = {}
        cds_count = 1

        with open(filename, 'r') as file:
            for record in SeqIO.parse(file, 'genbank'):
                for feature in record.features:
                    if feature.type == 'CDS':
                        location = feature.location
                        sequence = record.seq[location.start:location.end]
                        if location.strand == -1:  # Reverse strand
                            sequence = sequence.reverse_complement()
                        cds_sequences[f'CDS_{cds_count}'] = sequence
                        cds_count += 1

        return cds_sequences
    
    def gene_bank_file_creator_cds(self, tot_sequence_gene_bank_file_path: list):
        for file in tot_sequence_gene_bank_file_path:
            print(f'saving file:{file}')
            gene_id, molecular_type = self.extract_gene_info
            cds_sequence_dict = self.extract_cds_sequences(file)
            cds_track = []
            counter = 0
            for cds_x in cds_sequence_dict:
                print(cds_x)
                if counter == 0:
                    start = 1
                    stop = len(cds_sequence_dict[cds_x])
                    cds_sequence_dict[cds_x] = [cds_sequence_dict[cds_x], start, stop]
                else:   
                    start = cds_sequence_dict[cds_track[counter-1]][2]+1
                    stop = start + len(cds_sequence_dict[cds_x]) -1
                    cds_sequence_dict[cds_x] = [cds_sequence_dict[cds_x], start, stop]
                counter += 1
                cds_track.append(cds_x)
                
            #now we have the csd dic as cds_x : [start, stop, sequence]
            #so we can start making a new recod to save the sequence withoug introns as a genbank file
            record = SeqRecord(Seq(""), id=gene_id, annotations={"molecule_type": molecular_type})
            
            for cds_id, (sequence, start, end) in cds_sequence_dict.items():
                feature = SeqFeature(FeatureLocation(start-1, end), type="CDS", qualifiers={"gene_id": cds_id})
                record.features.append(feature)
                record.seq = record.seq + sequence
                
            # Write the record to a GenBank file
            with open(f"{gene_id}_cds.gb", "w") as output_handle:
                SeqIO.write(record, output_handle, "genbank")    
    
        return 


current_dir = os.getcwd()
db_path = os.path.join(current_dir, 'genome', 'gff.db')
fasta_path = os.path.join(current_dir, 'genome', 'primary_assembly_BY_260923_CI_02.fa')
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

explorer = GenomeExplorer(db_path, fasta_path, output_dir)


# gene_id = explorer.get_gene_info(identifier='KY.Chr14.256.v1.ND1-1_R0')
# gene_name = explorer.get_gene_info(identifier='Cint_N_Chr14_1002720.4')
# gene_ref = explorer.get_gene_info(identifier='KH.C14.432')

file_paths = explorer.gene_bank_file_creator_tot_sequence(identifier='KH.L108.56', raw = False)
# explorer
# cds = explorer.extract_cds_sequences(file_paths[0])


# explorer.gene_bank_file_creator_cds(file_paths)
