from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

import gffutils
import os 
from gffutils.exceptions import FeatureNotFoundError

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

import gffutils
import os 
from gffutils.exceptions import FeatureNotFoundError


def initialize(db_path, fasta_path):
    db = gffutils.FeatureDB(db_path)
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    return db, fasta_dict

def get_gene_by_name(db, name):
    feature_list = []
    for feature in db.all_features():
        gene_name = feature.attributes.get('Name')
        if gene_name and name in gene_name:
            feature_list.append(feature)
    return feature_list if feature_list else None

def get_gene_by_ref_gene(db, ref_gene_name):
    feature_list = []
    for feature in db.all_features():
        ref_gene = feature.attributes.get('ref-gene')
        if ref_gene and ref_gene_name in ref_gene:
            feature_list.append(feature)
    return feature_list if feature_list else None

def get_gene_info(db, identifier):
    try:
        gene = db[identifier]
    except FeatureNotFoundError:
        print(f"Failed to fetch gene by id: {identifier}")
        gene = get_gene_by_ref_gene(db, identifier)
        if gene is None:
            gene = get_gene_by_name(db, identifier)
        if gene is None:
            raise KeyError(f"No gene found with id, ref-gene, or name {identifier}")
    return gene

def get_gene_sequence(fasta_dict, gene):
    sequence = fasta_dict[gene.seqid].seq[gene.start - 1:gene.end]
    if gene.strand == '-':
        print('Gene on negative strand, reverse complementing sequence')
        sequence = sequence.reverse_complement()
        # for feature in db.children(gene.id):
        #     feature.start, feature.end = len(sequence) - feature.end, len(sequence) - feature.start
    return sequence

def get_cds_by_gene_gff_index(db, gene):
    """Obtain all CDS records associated with a gene
    GFF INDEXING STARTS AT 1, NOT 0
    """
    return list(db.children(gene.id, featuretype='CDS'))


def ajdust_cds_by_gene_gff_index(cds_by_gene_gff_index):
    """Used to adjust indexes to python indexing, relative or not to gene or genome"""
    #input is cds_by_gene_gff_index: Seqrecord of cds features with gff indexing relative to genome
    #cds_record_python_adjusted: Seqrecord with indexes relative to genome but with python indexing (for seq in cds_by_gene_gff_index seq[start -1, end-1])
    cds_record_python_adjusted = SeqRecord(sequence, id=gene.id, description="python index, genome index", annotations={"molecule_type": "DNA"})
    for cds in cds_by_gene_gff_index:
        cds_start = cds.start - 1
        cds_end = cds.end - 1
        cds_feature_python_adjusted = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_python_adjusted.features.append(cds_feature_python_adjusted)    
    #cds_record_gff_gene_to_start_adjusted = Seqrecord with indexes relative to gene start but with gff indexing (for seq in cds_by_gene_gff_index seq[start - gene.start + 1, end - gene.start + 1])     
    cds_record_gff_gene_to_start_adjusted = SeqRecord(sequence, id=gene.id, description="gff index, 1 for gene start", annotations={"molecule_type": "DNA"})
    for cds in cds_by_gene_gff_index:
        cds_start = cds.start - gene.start + 1
        cds_end = cds.end - gene.start + 1
        cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_gff_gene_to_start_adjusted.features.append(cds_feature)
    #cds_record_python_and_gene_to_start_adjusted = Seqrecord with indexes relative to gene start but with python indexing (for seq in cds_by_gene_gff_index seq[start - gene.start, end - gene.start])
    cds_record_python_and_gene_to_start_adjusted = SeqRecord(sequence, id=gene.id, description="python index, 0 for gene start", annotations={"molecule_type": "DNA"})
    for cds in cds_record_gff_gene_to_start_adjusted.features:
        cds_start = cds.location.start - 1
        cds_end = cds.location.end - 1
        cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_python_and_gene_to_start_adjusted.features.append(cds_feature) 
    return cds_record_python_adjusted, cds_record_gff_gene_to_start_adjusted, cds_record_python_and_gene_to_start_adjusted

def print_out_record_list(gene_cds_adjusted_index_list):
    for record in gene_cds_adjusted_index_list:
        print(f"Record ID: {record.id}")
        for feature in record.features:
            if feature.type == "CDS":
                print(f"CDS start: {feature.location.start}, CDS end: {feature.location.end}")
                
from Bio.SeqFeature import SeqFeature, FeatureLocation

def concatenated_cds_seq_from_seq_record_python_index_gene_ref(seq_record_python_index_gene_ref):
    """Function that concatenates the cds sequences of a seq_record with python indexing"""
    concatenated_cds_seq = ""
    cds_positions = []
    counter = 0
    for feature in seq_record_python_index_gene_ref.features:
        if feature.type == "CDS":
            cds_seq = seq_record_python_index_gene_ref.seq[feature.location.start:feature.location.end+1]
            print(f'extracted:start={feature.location.start};end={feature.location.end} ') #+1 because python is slicing exlusivley and gff inclusivley
            print(len(cds_seq))
            if counter == 0:
                print('first cds')
                cds_start = len(concatenated_cds_seq)
                counter += 1
            else:
                cds_start = len(concatenated_cds_seq)-1
            concatenated_cds_seq += str(cds_seq)
            cds_end = len(concatenated_cds_seq)-1
            cds_positions.append((cds_start, cds_end))
    concatenated_cds_seqrecord = SeqRecord(Seq(concatenated_cds_seq), id="new_id", description="Concatenated CDS sequences, gff index")

    #now we add the start end as feautres to the concatenated_cds_seqrecord
    for start, end in cds_positions:
        feature = SeqFeature(FeatureLocation(start, end), type="CDS")
        concatenated_cds_seqrecord.features.append(feature)
    concatenated_cds_seqrecord.annotations["molecule_type"] = "DNA"
    print(cds_positions)
    with open(f"{gene.id}_cds_concat.gb", "w") as output_handle:
        SeqIO.write(concatenated_cds_seqrecord, output_handle, "genbank")
    return concatenated_cds_seqrecord



def save_gene_record_tot_seq_genebank(seq_record_python_index_gene_ref):
    with open(f"{seq_record_python_index_gene_ref.id}_tot_seq.gb", "w") as output_handle:
        SeqIO.write(seq_record_python_index_gene_ref, output_handle, "genbank")


#####
  
current_dir = os.getcwd()
db_path = os.path.join(current_dir, 'genome', 'gff.db')
fasta_path = os.path.join(current_dir, 'genome', 'primary_assembly_BY_260923_CI_02.fa')
output_dir = os.path.join(current_dir, 'output')

###Initiation build db and fasta
fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
db = gffutils.FeatureDB(db_path)


#search for gene in gff to obtain coordinates
identifier='KY.Chr2.2230'
genes = get_gene_info(db, identifier)
gene = genes[0]

#get fasta sequence
sequence = get_gene_sequence(fasta_dict, gene)
#build a record for the sequence
record = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
#get cds features, gff indexing!
gene_cds_gff_index = get_cds_by_gene_gff_index(db, gene)
gene_cds_gff_index_list = ajdust_cds_by_gene_gff_index(gene_cds_gff_index)
print_out_record_list(gene_cds_gff_index_list)

seq_record_python_index_gene_ref = gene_cds_gff_index_list[2]
save_gene_record_tot_seq_genebank(seq_record_python_index_gene_ref)
concatenated_cds_seq_from_seq_record_python_index_gene_ref(seq_record_python_index_gene_ref)