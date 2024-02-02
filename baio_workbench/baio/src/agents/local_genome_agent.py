
from src.non_llm_tools.genome_explorer import GeneInfoExtractor, GeneAnalyzer
from Bio import SeqIO
###test
# current_dir = os.getcwd()

# # # Define the paths to the GFF database and the FASTA file
# db_path = os.path.join(current_dir, 'genome', 'gff.db')
# fasta_path = os.path.join(current_dir, 'genome', 'primary_assembly_BY_260923_CI_02.fa')
# output_dir = os.path.join(current_dir, 'output/')

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
# gene_analyzer = GeneAnalyzer(db_path, fasta_path, gene, output_dir)
# gene_analyzer
# # Get the sequence of the gene from the FASTA file

# # Create a SeqRecord object for the gene
# # Get the CDS features of the gene from the GFF database
# gene_cds_gff_index = gene_analyzer.get_cds_by_gene_gff_index(gene)
# data_dict = gene_analyzer.cds_coordinates_to_dict()
# rec_conc = gene_analyzer.concatenate_cds_sequences()
# gene_analyzer.add_features_to_record('coordinate_type_gff_gene_adjusted')
# record = gene_analyzer.record
# df = gene_analyzer.cds_coordinates_to_df_save_csv()
# record = gene_analyzer.record

# # Save rec_conc as a GenBank file
# with open("rec_conc.gb", "w") as output_handle:
#     SeqIO.write(rec_conc, output_handle, "genbank")

# # Save record as a GenBank file
# with open("record.gb", "w") as output_handle:
#     SeqIO.write(record, output_handle, "genbank")


def genome_explorer(identifier:str, genome_db_path, genome_fasta_path, output_dir):

    analyzer = GeneInfoExtractor(genome_db_path, genome_fasta_path)
    analyzer.initialize()
    analyzer.gene = identifier
    genes = analyzer.gene
    counter = 0
    for gene in genes:
        gene_analyzer = GeneAnalyzer(genome_db_path, genome_fasta_path, gene, output_dir)

        gene_cds_gff_index = gene_analyzer.get_cds_by_gene_gff_index(gene)
        data_dict = gene_analyzer.cds_coordinates_to_dict()
        rec_conc = gene_analyzer.concatenate_cds_sequences()
        # rec_conc = gene_analyzer.add_features_to_record('coordinate_type_gff_gene_adjusted')
        # rec_conc = gene_analyzer.add_features_to_record('coordinate_type_gff_gene_adjusted')

        gene_analyzer.add_features_to_record('coordinate_type_gff_gene_adjusted')
        record = gene_analyzer.record
        df = gene_analyzer.cds_coordinates_to_df_save_csv()
        record = gene_analyzer.record
        print(df)
        print(record)
        # Save rec_conc as a GenBank file
        with open(f"{output_dir}/{gene.id}_concatenated_cds.gb", "w") as output_handle:
            SeqIO.write(rec_conc, output_handle, "genbank")

        # Save record as a GenBank file
        with open(f"{output_dir}/{gene.id}_v{counter}.gb", "w") as output_handle:
            SeqIO.write(record, output_handle, "genbank")
        df.to_csv(f'{output_dir}{gene.id}_coordinates_v{counter}.csv', index=True)
