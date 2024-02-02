from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import gffutils
import os 


class GenomeExplorer:
    def __init__(self, db_path, fasta_path, output_dir):
        self.db = gffutils.FeatureDB(db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        self.output_dir = output_dir
        
    def get_gene_info(self, gene_id):
        # Fetch gene from the database
        gene = self.db[gene_id]
        return gene

    def get_gene_sequence(self, gene_id):
        # Fetch sequence from the FASTA file
        gene = self.get_gene_info(gene_id)
        sequence = self.fasta_dict[gene.seqid].seq[gene.start - 1:gene.end]
        return sequence

    def create_genbank_file(self, gene_id):
        # Fetch gene and sequence
        gene = self.get_gene_info(gene_id)
        print(f'Gene: {gene}\n')
        sequence = self.get_gene_sequence(gene_id)
        # print(sequence)
        # Create a SeqRecord
        record = SeqRecord(sequence, id=gene_id, description="", annotations={"molecule_type": "DNA"})
        print(f'record:{record}')
        # Add gene feature to the record
        feature = SeqFeature(FeatureLocation(gene.start - 1, gene.end), type="gene", qualifiers={"gene_id": gene_id})
        record.features.append(feature)
        print(feature)
        # Construct the output path
        output_path = os.path.join(self.output_dir, f'{gene_id}.gb')

        # Write the record to a GenBank file
        SeqIO.write([record], output_path, "genbank")

# Get the current working directory
current_dir = os.getcwd()

# Construct the paths to the database and FASTA file
db_path = os.path.join(current_dir, 'genome', 'gff.db')
fasta_path = os.path.join(current_dir, 'genome', 'primary_assembly_BY_260923_CI_02.fa')
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
explorer = GenomeExplorer(db_path, fasta_path, output_dir)

gene_id = "KY.UAContig13.25.v1.ND1-1_R5"
record = explorer.get_gene_info(gene_id)


# sequence = explorer.get_gene_sequence(gene_id)
# explorer.create_genbank_file(gene_id)
# gene = explorer.get_gene_info(gene_id)
def get_gene_by_ref_gene(db, ref_gene_name):
    """Obtain the gene record based on ref-gene name"""
    for feature in db.all_features():
        ref_gene = feature.attributes.get('ref-gene')
        if ref_gene and ref_gene_name in ref_gene:
            return feature
    return None


# Use the function like this:
db = gffutils.FeatureDB(db_path)
def get_cds_by_gene(db, gene_id):
    cds_records = []
    for cds in db.all_features(featuretype='CDS'):
        if 'Parent' in cds.attributes and gene_id in cds.attributes['Parent']:
            print(cds_records)
            cds_records.append(cds)
    return cds_records

gene = get_gene_by_ref_gene(db, 'KH.C14.432')

cds_records = get_cds_by_gene(db, gene.id)

fasta = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))


cds_records = get_cds_by_gene(db, 'KY.UAContig32.4')

for cds in cds_records:
    print(f"ID: {cds.id}")
    print(f"Start: {cds.start}")
    print(f"End: {cds.end}")
    print(f"Strand: {cds.strand}")
    print(f"Attributes: {cds.attributes}")
    print(f"Sequence: {cds.sequence}")
    
explorer = GenomeExplorer(db_path, fasta_path, output_dir)
explorer.create_genbank_file(gene_id=gene.id)

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq

# Assuming 'gene' is your gene feature and 'cds_records' is a list of CDS features
# Create a SeqRecord
record = SeqRecord(Seq(""), id=gene.id, description="")

# Add gene feature to the record
strand = 1 if gene.strand == '+' else -1 if gene.strand == '-' else 0
gene_feature = SeqFeature(FeatureLocation(start=gene.start, end=gene.end, strand=strand), type="gene")
record.features.append(gene_feature)

# Add CDS features to the record
for cds in cds_records:
    cds_feature = SeqFeature(FeatureLocation(start=cds.start, end=cds.end, strand=cds.strand), type="CDS")
    record.features.append(cds_feature)

# Write the record to a GenBank file
SeqIO.write([record], "output.gb", "genbank")

cds_records = list(db.children(gene.id, featuretype='CDS'))
# Create a SeqRecord
record = SeqRecord(Seq(""), id=gene.id, description="")

# Add molecule_type to the record's annotations
record.annotations["molecule_type"] = gene.featuretype  # 'mRNA'

# Rest of your code...

# Add gene feature to the record
strand = 1 if gene.strand == '+' else -1 if gene.strand == '-' else 0
gene_feature = SeqFeature(FeatureLocation(start=gene.start, end=gene.end, strand=strand), type="gene")
record.features.append(gene_feature)

for cds in cds_records:
    strand = 1 if cds.strand == '+' else -1 if cds.strand == '-' else 0
    cds_feature = SeqFeature(FeatureLocation(start=cds.start, end=cds.end, strand=strand), type="CDS")
    record.features.append(cds_feature)

# Get the start position of the gene
gene_start = record.features[0].location.start

# Subtract the start position of the gene from the start and end positions of each feature
for feature in record.features:
    feature.location = FeatureLocation(start=feature.location.start - gene_start,
                                       end=feature.location.end - gene_start ,
                                       strand=feature.location.strand)

# Write the record to a GenBank file

record.seq = explorer.get_gene_sequence('KH.C14.432.v1.B.ND1-1_R0')
# Write the record to a GenBank file
SeqIO.write([record], "output.gb", "genbank")

record_gene = explorer.build_record(gene_id='KY.UAContig13.25.v1.ND1-1_R5')



from Bio import SeqIO, SeqFeature, SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

# Fetch a gene by its identifier
gene = explorer.get_gene_info(identifier='KY.Chr14.256.v1.ND1-1_R0')

# Get the CDS features of the gene

sequence = fasta[gene.seqid].seq[gene.start - 1:gene.end]
record = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
gene_feature = SeqFeature(FeatureLocation(0, gene.end-gene.start), type="gene", qualifiers={"gene_id": gene.id})
record.features.append(gene_feature)
cds_records = list(db.children(gene.id, featuretype='CDS'))
for cds in cds_records:
    # Adjust the start and end locations of the CDS feature
    # keep the -1 and +1 for clarification of the indexing adjustment
    cds_start = cds.start -1 - gene.start + 1
    cds_end = cds.end -1 - gene.start + 1

    # Create a SeqFeature for the CDS
    cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
    record.features.append(cds_feature)
output_path = os.path.join(output_dir, f'{gene.id}.gb')
SeqIO.write([record], output_path, "genbank")
print(f'{gene.id}.gb')

cds_counter = 1

# Initialize dictionary to store CDS sequences
cds_dict = {}

for cds in cds_records:
    # Adjust the start and end locations of the CDS feature
    cds_start = cds.start -1 - gene.start + 1
    cds_end = cds.end -1 - gene.start + 1

    # Extract the CDS sequence
    cds_sequence = fasta[gene.seqid].seq[cds_start:cds_end]

    # Store the CDS sequence in the dictionary
    cds_dict[f'CDS_{cds_counter}'] = cds_sequence

    # Increment the CDS counter
    cds_counter += 1

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import SeqIO

# Initialize list to store SeqRecord objects
records = []

# Iterate over items in cds_dict
for key, sequence in cds_dict.items():
    # Create SeqRecord object
    record = SeqRecord(Seq(str(sequence)), id=key, description="", annotations={"molecule_type": "DNA"})

    # Create SeqFeature object
    feature = SeqFeature(FeatureLocation(0, len(sequence)), type="CDS")

    # Add SeqFeature object to SeqRecord object's features list
    record.features.append(feature)

    # Append SeqRecord object to list
    records.append(record)

# Write list of SeqRecord objects to GenBank file
SeqIO.write(records, "output.gb", "genbank")


from Bio import SeqIO

def extract_cds_sequences(filename):
    cds_sequences = {}
    cds_count = 1

    with open(filename, 'r') as file:
        for record in SeqIO.parse(file, 'genbank'):
            for feature in record.features:
                if feature.type == 'CDS':
                    location = feature.location
                    sequence = record.seq[location.start:location.end]
                    cds_sequences[f'CDS_{cds_count}'] = sequence
                    cds_count += 1

    return cds_sequences

file_name= "KH.L108.56.v1.B.nonSL2-1_R0_raw.gb"
output_path=f'/usr/src/app/biosnaker-genome/output/{file_name}'
seq_dic = extract_cds_sequences(output_path)



cds_track = []
counter = 0
for cds_x in seq_dic:
    print(cds_x)
    if counter == 0:
        start = 1
        stop = len(seq_dic[cds_x])
        seq_dic[cds_x] = [seq_dic[cds_x], start, stop]
    else:   
        start = seq_dic[cds_track[counter-1]][2]+1
        stop = start + len(seq_dic[cds_x]) -1
        seq_dic[cds_x] = [seq_dic[cds_x], start, stop]
    counter += 1
    cds_track.append(cds_x)

    # print(len(seq_dic[cds_x]))
    print('\n')
    
    # Create an empty SeqRecord
record = SeqRecord(Seq(""), id=file_name, annotations={"molecule_type": "DNA"})

# Add CDS features to the record
for cds_id, (sequence, start, end) in seq_dic.items():
    feature = SeqFeature(FeatureLocation(start-1, end), type="CDS", qualifiers={"gene_id": cds_id})
    record.features.append(feature)
    record.seq = record.seq + sequence

# Write the record to a GenBank file
with open(f"{file_name}_con.gb", "w") as output_handle:
    SeqIO.write(record, output_handle, "genbank")
    

cds = explorer.extract_cds_sequences(file_paths[0])
extract_cds_sequences(file_paths[0])

########
########

# Rebuilding from scratch
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
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord

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
        for feature in db.children(gene.id):
            feature.start, feature.end = len(sequence) - feature.end, len(sequence) - feature.start
    return sequence

def get_cds_by_gene_gff_index(db, gene):
    """Obtain all CDS records associated with a gene
    GFF INDEXING STARTS AT 1, NOT 0
    """
    return list(db.children(gene.id, featuretype='CDS'))


    
def create_genbank_file(db, output_dir, get_gene_info, get_gene_sequence, identifier, raw):
    """
    Create a GenBank file for a gene containing CDS features.

    Args:
        db: The database object.
        output_dir (str): The output directory.
        get_gene_info (function): Function to get gene info.
        get_gene_sequence (function): Function to get gene sequence.
        identifier (str): Identifier of the gene.
        raw (bool): If False, adjust CDS start and end locations.
    """
    generated_file_paths = []
    genes = get_gene_info(identifier)
    print(f'Found {len(genes)} genes matching the identifier {identifier}')
    for gene in genes:
        sequence = get_gene_sequence(gene)
        record = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
        gene_feature = SeqFeature(FeatureLocation(0, gene.end - gene.start), type="gene", qualifiers={"gene_id": gene.id})
        record.features.append(gene_feature)
        cds_records = list(db.children(gene.id, featuretype='CDS'))
        if not raw:
            print('Adjusting CDS start and end locations')
            for cds in cds_records:
                cds_start = cds.start - gene.start + 1                
                cds_end = cds.end - gene.start + 1
                cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
                record.features.append(cds_feature)
            output_path = os.path.join(output_dir, f'{gene.id}.gb')
            generated_file_paths.append(output_path)
            SeqIO.write([record], output_path, "genbank")
            print(f'\nFile: {gene.id}.gb\nSaved in: {output_path}\n')
    return generated_file_paths


###Initiation build db and fasta
fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

#search for gene in gff to obtain coordinates
identifier='KH.L108.56'
genes = get_gene_info(db, identifier)
gene = genes[0]

#get fasta sequence
sequence = get_gene_sequence(fasta_dict, gene)
#build a record for the sequence
record = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})

#add gene feature to the record
#python indexing: feautre starts at 0!
gene_feature = SeqFeature(FeatureLocation(0, len(sequence)-1), type="gene", qualifiers={"gene_id": gene.id})

#get cds features, gff indexing!
gene_cds = get_cds_by_gene_gff_index(db, gene)

#now we have to reindex the cds features to match the sequence, so start at 0 and end at len(sequence)-1
#python indexing: feautre starts at 0!
##for first cds:
cds_0_stop = gene_cds[0].stop - gene_cds[0].start -1 #-1 to adjust for python indexing
##because we are always relative to the cds_0_start we always subtract cds_0_start -1 from the cds cooridinates we are interested in 

###
###logic explained:
#we set the first nct of the sequence as the first index, for this we subtract the start index of the gene from the cds start index
# a = '123456789'
a_gff_index = {'start': 10, 'stop': 20}
# +1 for each index because we are dealing with INCLUSIVE coordinates
a_gff_gene_adjusted_index = {'start' : a_gff_index['start'] - a_gff_index['start'] +1 , 'stop': a_gff_index['stop'] - a_gff_index['start']+1}
# a_gff_gene_adjusted_index = {'start': 1, 'stop': 20}
a_python_adjusted_index = {'start': a_gff_index['start'] -1, 'stop': a_gff_index['stop'] - 1}
a_python_gene_adjusted_index = {'start': a_gff_gene_adjusted_index['start'] - 1, 'stop': a_gff_gene_adjusted_index['stop'] - 1}

###logic for the 
#now we apply the same logic to the cds features
record_adjusted = SeqRecord(sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
cdss_adjusted = []
for cds in gene_cds:
    cds_start = cds.start - gene.start + 1
    cds_end = cds.end - gene.start + 1
    cds_feature_adjusted = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
    cdss_adjusted.append(cds_feature_adjusted)
    record_adjusted.features.append(cds_feature_adjusted)

def ajdust_cds_by_gene_gff_index(cds_by_gene_gff_index):
    """Used to adjust indexes to python indexing"""
    #cds_record_python_adjusted: Seqrecord with indexes relative to genome
    cds_record_gff_gene_to_start_adjusted = SeqRecord(sequence, id=gene.id, description="gff index, 1 for gene start", annotations={"molecule_type": "DNA"})
    for cds in cds_by_gene_gff_index:
        cds_start = cds.start - gene.start + 1
        cds_end = cds.end - gene.start + 1
        cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_gff_gene_to_start_adjusted.features.append(cds_feature)

    cds_record_python_and_gene_to_start_adjusted = SeqRecord(sequence, id=gene.id, description="python index, 1 for gene start", annotations={"molecule_type": "DNA"})
    for cds in cds_record_gff_gene_to_start_adjusted.features:
        cds_start = cds.location.start - 1
        cds_end = cds.location.end - 1
        cds_feature = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_python_and_gene_to_start_adjusted.features.append(cds_feature)
        
    cds_record_python_adjusted = SeqRecord(sequence, id=gene.id, description="python index, genome index", annotations={"molecule_type": "DNA"})
    for cds in cds_by_gene_gff_index:
        cds_start = cds.start - 1
        cds_end = cds.end - 1
        cds_feature_python_adjusted = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
        cds_record_python_adjusted.features.append(cds_feature_python_adjusted)    

# cds_record_python_adjusted = SeqRecord(sequence, id=gene.id, description="python index, genome index", annotations={"molecule_type": "DNA"})
# for cds in cds_by_gene_gff_index:
#     cds_start = cds.start - 1
#     cds_end = cds.end - 1
#     cds_feature_python_adjusted = SeqFeature(FeatureLocation(cds_start, cds_end), type="CDS")
#     cds_record_python_adjusted.features.append(cds_feature_python_adjusted)
#cds_record_python_and_gene_to_start_adjusted: Seqrecord with indexes relative to gene start     

#def extract_coding_sequence_of_cds_by_gene(cds_record_gene_to_start_adjusted):
gene_cds_gff_index_list = ajdust_cds_by_gene_gff_index(gene_cds_gff_index)
for cds in gene_cds_gff_index_list[-1].features:
    cds_start = cds.location.start
    print(cds_start)
    cds_end = cds.location.end
    print(cds_end)
    cds_sequence = gene_cds_gff_index_list[-1].seq[cds_start:cds_end]
    cds_record = SeqRecord(cds_sequence, id=gene.id, description="", annotations={"molecule_type": "DNA"})
    cds_record.features.append(cds)
        
        
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
###
###     EXTRACT THE CDS OF THE GFF INDEXED AT GENE START SEQUENCE
###
concatenated_cds_seq = ""
cds_positions = []
counter = 0
for feature in gene_cds_gff_index_list[1].features:
    if feature.type == "CDS":
        print('in cds')
        cds_seq = gene_cds_gff_index_list[-1].seq[feature.location.start:feature.location.end]
        if counter == 0:
            print('first cds')
            cds_start = 1
            cds_end = len(cds_seq)
            counter = 1
            print(f'cds_start: {cds_start}, cds_end: {cds_end}')
        else:
            cds_start = len(concatenated_cds_seq) + 1
            cds_end = len(concatenated_cds_seq)
        concatenated_cds_seq += str(cds_seq)
        cds_positions.append((cds_start, cds_end))
concatenated_cds_seqrecord = SeqRecord(Seq(concatenated_cds_seq), id="new_id", description="")
print('gff index:')
print(cds_positions)

####    FOR python INDEXING, STARTING at 0 for nct at position 1
from Bio.SeqFeature import SeqFeature, FeatureLocation

seq_record_python_index_gene_ref = gene_cds_gff_index_list[2]
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
concatenated_cds_seqrecord = concatenated_cds_seq_from_seq_record_python_index_gene_ref(seq_record_python_index_gene_ref)

def save_gene_record_tot_seq_genebank(seq_record_python_index_gene_ref):
    with open("f{seq_record_python_index_gene_ref.id}_tot_seq.gb", "w") as output_handle:
        SeqIO.write(seq_record_python_index_gene_ref, output_handle, "genbank")



####    FOR GFF INDEXING, STARTING at 1 for nct at position 1
concatenated_cds_seq = ""
cds_positions = []
seq_record = gene_cds_gff_index_list[2]
for feature in seq_record.features:
    if feature.type == "CDS":
        cds_seq = seq_record.seq[feature.location.start:feature.location.end]
        print(feature.location.start)

        print(feature.location.end)
        print(cds_seq)
        cds_start = len(concatenated_cds_seq)
        concatenated_cds_seq += str(cds_seq)
        cds_end = len(concatenated_cds_seq)-1
        cds_positions.append((cds_start, cds_end))
        
concatenated_cds_seqrecord = SeqRecord(Seq(concatenated_cds_seq), id="new_id", description="Concatenated CDS sequences, python index")

sum(len(feature) for feature in seq_record.features if feature.type == "CDS")        

>>> cds_positions
[(1, 180), (181, 534), (535, 668), (669, 827)]     
        
        
        
        
        
        
        
        
total_cds_length = sum(len(feature) for feature in gene_cds_gff_index_list[-1].features if feature.type == "CDS")
#for the end index of the cds we subtract the cds.end 
cds_feature_adjusted = SeqFeature(FeatureLocation(cds.start - gene.start, cds_0_stop), type="CDS")

gene_cds2[0].start
##equivalent with biopython
cds_features_not_adjusted = [SeqFeature(FeatureLocation(cds.start, cds.end), type="CDS") for cds in gene_cds]
cds_features_adjusted = [SeqFeature(FeatureLocation(cds.start - 1, cds.end -1), type="CDS") for cds in gene_cds]


len_cds_1 = gene_cds2[0].stop - gene_cds2[0].start
print(f'Length of CDS 1 from gff object: {len_cds_1} from {gene_cds2[0].start} to {gene_cds2[0].stop}')
cds_1_not_adjsuted_length = cds_features_not_adjusted[0].location.end - cds_features_not_adjusted[0].location.start
print(f'Length of CDS 1 from unadjusted biopython object: {cds_1_not_adjsuted_length} from {gene_cds2[0].start} to {gene_cds2[0].stop}')

#for each cds adjust index by:
cds_


######
######      29.01.24
######

#checking the sequence 
from Bio.Seq import Seq

from Bio.Seq import Seq

def find_sequence_in_contig(fasta_dict, contig, query_sequence):
    contig_sequence = str(fasta_dict[contig].seq)
    query_sequence_complement = str(Seq(query_sequence).complement())
    query_sequence_rev_complement = str(Seq(query_sequence).reverse_complement())
    
    for sequence, description in [(query_sequence, "direct"), (query_sequence_complement, "complement"), (query_sequence_rev_complement, "reverse complement")]:
        start = contig_sequence.find(sequence)
        if start != -1:  # if the sequence was found
            end = start + len(sequence)
            print(f"Found {description} sequence at positions {start} to {end} in contig {contig}")
            return start, end

    print(f"Neither sequence, its complement, nor its reverse complement were found in contig {contig}")
    return None, None


seq_record_gene = fasta_dict[gene.seqid]

contig = fasta_dict[gene.seqid].id
query_sequence = "TTTATCAAACACATAAGA"

start, end = find_sequence_in_contig(fasta_dict, contig, query_sequence)




def print_out_record_list(gene_cds_adjusted_index_list, fasta_dict):
    data = []
    for record in gene_cds_adjusted_index_list:
        for feature in record.features:
            if feature.type == "CDS":
                if record.description == "Python adjusted CDS record with genome indexing":
                    seq = fasta_dict[record.id].seq
                else:
                    seq = record.seq
                if feature.location.start < len(seq) and feature.location.end <= len(seq):
                    data.append({
                        'indexing_type': record.description,
                        'start_index': feature.location.start,
                        'start_seq': seq[feature.location.start],
                        'end_index': feature.location.end,
                        'end_seq': seq[feature.location.end - 1],  # -1 because end is exclusive
                        'length': len(feature)
                    })
                else:
                    print(f"Feature {feature} in record {record.id} has indices outside the range of the sequence")
    df = pd.DataFrame(data)
    print(df)
    return df
    
df = print_out_record_list([gene_cds_gff_index_list[0]], fasta_dict)

record = gene_cds_gff_index_list[0].id