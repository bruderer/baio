import os
from copy import deepcopy

import gffutils
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord
from gffutils import Feature


class CustomFeature(Feature):
    def __init__(self, seqid=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqid = seqid


class GeneInfoExtractor:
    """Class to retrieve a specific gene feature from a gff and fasta file.
    It will check ref-gene, id, and name attributes to find the gene."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.db = gffutils.FeatureDB(db_path)
        self._gene = None  # Initialize the _gene attribute
        self.id = None  # Initialize the id attribute
        self.version = (
            "0.1"  # should be the genome version from which the gene comes from
        )

    @property
    def gene(self):
        return self._gene

    def get_gene_by_name(self, name):
        feature_list = []
        for feature in self.db.all_features():
            gene_name = feature.attributes.get("Name")
            if gene_name and name in gene_name:
                feature_list.append(feature)
        return feature_list if feature_list else None

    def get_gene_by_ref_gene(self, ref_gene_name):
        feature_list = []
        for feature in self.db.all_features():
            ref_gene = feature.attributes.get("ref-gene")
            if ref_gene and ref_gene_name in ref_gene:
                feature_list.append(feature)
        return feature_list if feature_list else None

    @gene.setter
    def gene(self, identifier):
        try:
            self._gene = self.db[identifier]
            self.id = self._gene.id  # Set the id attribute
        except gffutils.FeatureNotFoundError:
            print(f"Failed to fetch gene by id: {identifier}")
            self._gene = self.get_gene_by_ref_gene(identifier)
            if self._gene is None:
                self._gene = self.get_gene_by_name(identifier)
            if self._gene is None:
                raise KeyError(f"No gene found with id, ref-gene, or name {identifier}")


class SequenceExtractor:
    """
    A class for extracting gene sequences using Python indexing.

    Args:
        db_path (str): The path to the GFF database file.
        fasta_path (str): The path to the FASTA file.
        gene (str or gffutils.feature.Feature): The gene identifier or GFF feature.
        start (int, optional): The start position of the gene sequence PYTHON INDEXING.
        Defaults to None.
        end (int, optional): The end position of the gene sequence PYTHON INDEXING.
        Defaults to None.
        strand (str, optional): The strand of the gene sequence. Defaults to None.

    Attributes:
        db (gffutils.FeatureDB): The GFF database.
        fasta_dict (dict): A dictionary containing the sequences from the FASTA file.
        gene (str or gffutils.feature.Feature): The gene identifier or GFF feature.
        start (int or None): The start position of the gene sequence.
        end (int or None): The end position of the gene sequence.
        sequence (Bio.Seq.Seq or None): The extracted gene sequence.

    Methods:
        get_gene_sequence: Extracts the gene sequence based on the provided parameters.

    """

    def __init__(self, db_path, fasta_path, seqid, start=None, end=None):
        self.db = gffutils.FeatureDB(db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        self.seqid = seqid
        self.start = start
        self.end = end
        self.sequence = None
        # self.features = features
        self.sequence = self.get_gene_sequence()

    def get_gene_sequence(self):
        """
        Extracts the gene sequence based on the provided parameters.

        Returns:
            Bio.Seq.Seq: The extracted gene sequence.

        """
        PYTHON_SLICE_OFFSET = 1

        sequence = self.fasta_dict[self.seqid].seq[
            self.start - PYTHON_SLICE_OFFSET : self.end
        ]

        return sequence


class GeneComponentExtractor:
    """Stores cds of a gene"""

    def __init__(self, db_path, fasta_path, gene):
        """
        Initializes a GeneComponentExtractor object.

        Args:
            db_path (str): The path to the GFF database file.
            fasta_path (str): The path to the FASTA file.
            gene (gffutils.Feature): The gene feature to extract components for.
        """
        self.db = gffutils.FeatureDB(db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        self.seqid = gene.seqid
        self.gene = gene
        # self.total_contig_length = total_contig_length
        self.features = list(self.db.children(self.gene.id))
        # self.features = sorted(self.features, key=lambda feature: feature.location.start)
        self.cds = list(self.db.children(self.gene.id, featuretype="CDS"))


class gffutilsFeatureToSeqFeature:
    """
    Transforms a gffutils.feature.Feature object to a Bio.SeqFeature.SeqFeature object.

    Args:
        feature (gffutils.feature.Feature): The feature to transform.

    Returns:
        Bio.SeqFeature.SeqFeature: The transformed feature.
    """

    def __init__(self, feature):
        self.feature = feature
        self.seqfeature = self.transform_feature()

    def transform_feature(self):
        """
        Transforms the feature to a Bio.SeqFeature.SeqFeature object.

        Returns:
            Bio.SeqFeature.SeqFeature: The transformed feature.
        """
        return SeqFeature(
            FeatureLocation(self.feature.start, self.feature.end),
            type=self.feature.featuretype,
            strand=self.feature.strand,
        )


class RawInfoUpDater:
    """Make your raw_info ready to be saved as genebank file
    Transforms a sequence and features to its reverse complement.
    Transforms gffutils features into bio.seqfeature objects.
    features_gene_centric: always the updated ones!"""

    def __init__(self, total_contig_length, sequence, raw_info):
        self.sequence = sequence.sequence

        self.raw_info = raw_info
        self.total_contig_length = total_contig_length
        self.PYTHON_OFFSET = 1

    def run_me(self):
        if self.raw_info.gene.strand == "-":
            print(
                "Processing reverse complement: updating and re-indexing features to "
                "gene centric and genome centric"
            )
            self.updated_sequence = self.transform_to_reverse_complement()
            self.features_gene_centric = (
                self.transform_features_to_reverse_complement_gene_centric()
            )
            self.features_genome_centric = (
                self.transform_features_to_reverse_complement_genome_centric()
            )
        if self.raw_info.gene.strand == "+":
            print(
                "Processing forward sequence: updating features to gene centric and genome centric"
            )
            self.updated_sequence = self.sequence
            self.features_gene_centric = self.transform_features_to_gene_centric()
            self.features_genome_centric = self.transform_features_to_genome_centric()

    def transform_to_reverse_complement(self):
        """
        Transforms the sequence to its reverse complement.

        Returns:
            Bio.Seq.Seq: The reverse complement of the sequence.
        """
        return self.sequence.reverse_complement()

    def transform_features_to_reverse_complement_gene_centric(self):
        """
        Transforms the features to their reverse complement.
        Strand will be changed from + to -
        Args:
            features (list): A list of gffutils.feature.Feature objects.

        Returns:
            list: A list of gffutils.feature.Feature objects with reverse complemented coordinates.

        """
        reverse_complement_features = []
        i = len(self.raw_info.cds)
        for feature in self.raw_info.cds:
            reverse_complement_feature = deepcopy(feature)
            reverse_complement_feature.start = self.raw_info.gene.end - feature.end
            reverse_complement_feature.end = (
                self.raw_info.gene.end - feature.start + self.PYTHON_OFFSET
            )
            reverse_complement_feature.strand = 1
            reverse_complement_feature.type = (
                feature.featuretype
            )  # Preserve the type attribute
            reverse_complement_feature = gffutilsFeatureToSeqFeature(
                reverse_complement_feature
            ).seqfeature
            reverse_complement_feature.qualifiers["label"] = ["CDS_{}".format(i)]
            reverse_complement_features.append(reverse_complement_feature)

            i -= 1

        return reverse_complement_features

    def transform_features_to_reverse_complement_genome_centric(self):
        """
        Transforms the features to their reverse complement.
        Strand will be changed from + to -
        Args:
            features (list): A list of gffutils.feature.Feature objects.

        Returns:
            list: A list of gffutils.feature.Feature objects with reverse complemented coordinates.

        """
        reverse_complement_features = []
        i = len(self.raw_info.cds)
        for feature in self.raw_info.cds:
            reverse_complement_feature = deepcopy(feature)
            reverse_complement_feature.start = self.total_contig_length - feature.end
            reverse_complement_feature.end = (
                self.total_contig_length - feature.start + self.PYTHON_OFFSET
            )
            reverse_complement_feature.strand = 1
            reverse_complement_feature.type = "CDS"
            reverse_complement_feature.type = (
                feature.featuretype
            )  # Preserve the type attribute
            reverse_complement_feature = gffutilsFeatureToSeqFeature(
                reverse_complement_feature
            ).seqfeature
            reverse_complement_feature.qualifiers["element"] = ["CDS_{}".format(i)]
            reverse_complement_features.append(reverse_complement_feature)
            i -= 1

        return reverse_complement_features

    def transform_features_to_genome_centric(self):
        """
        Convert the gff utils features to SeqFeatures
        Args:
            features (list): A list of gffutils.feature.Feature objects.

        Returns:
            list: A list of SeqFeatures objects with reverse complemented coordinates.

        """
        features = []
        i = 1
        for feature in self.raw_info.cds:
            up_dated_feature = deepcopy(feature)
            up_dated_feature.start = feature.start
            up_dated_feature.end = feature.end
            up_dated_feature.strand = 1
            up_dated_feature.type = "CDS"
            up_dated_feature.type = feature.featuretype  # Preserve the type attribute
            up_dated_feature = gffutilsFeatureToSeqFeature(up_dated_feature).seqfeature
            up_dated_feature.qualifiers["element"] = ["CDS_{}".format(i)]
            features.append(up_dated_feature)
            i += 1
        return features

    def transform_features_to_gene_centric(self):
        features = []
        i = 1
        for feature in self.raw_info.cds:
            up_dated_feature = deepcopy(feature)
            up_dated_feature.start = feature.start - self.raw_info.gene.start
            up_dated_feature.end = feature.end - self.raw_info.gene.start
            up_dated_feature.strand = 1
            up_dated_feature.type = feature.featuretype  # Preserve the type attribute
            up_dated_feature = gffutilsFeatureToSeqFeature(up_dated_feature).seqfeature
            up_dated_feature.qualifiers["element"] = ["CDS_{}".format(i)]
            features.append(up_dated_feature)
            i += 1
        return features


class Concatenater:
    """Concatenate the features and the sequence"""

    def __init__(self, sequence, features):
        self.features = deepcopy(
            sorted(features, key=lambda feature: feature.location.start)
        )
        self.concatenated_sequence = Seq(
            deepcopy(
                "".join([str(feature.extract(sequence)) for feature in self.features])
            )
        )
        self.concatenated_features = self.reindex_features()

    def reindex_features(self):
        OFFSET = 1
        reindexed_features = []
        cumulative_length = 0
        for feature in self.features:
            feature_length = len(feature)
            reindexed_feature = feature
            if cumulative_length == 0:
                reindexed_feature.location = FeatureLocation(
                    cumulative_length,
                    cumulative_length + feature_length,
                    strand=feature.location.strand,
                )
            else:
                reindexed_feature.location = FeatureLocation(
                    cumulative_length,
                    cumulative_length + feature_length,
                    strand=feature.location.strand,
                )
            reindexed_features.append(reindexed_feature)
            cumulative_length += feature_length
        return reindexed_features
        # self.concatenated_sequence = self.concatenate_sequence()


def make_genbank_file(
    id, name, description, sequence, features, molecule_type, output_dir, suffix=""
):
    record = SeqRecord(
        sequence, id=id, name=name, description=description, features=features
    )
    record.annotations["molecule_type"] = molecule_type
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + id + "_" + suffix + ".gb", "w") as output_handle:
        SeqIO.write(record, output_handle, "genbank")


class GenomeRegionExtractor:
    def __init__(self, db_path, fasta_path, start, end, seqid, strand):
        self.db = gffutils.FeatureDB(db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        self.seqid = seqid
        self.start = start
        self.end = end
        self.region_features = None
        self.strand = strand

    def get_features_in_region(self, start, end):
        return self.db.region((self.seqid, start, end))

    def region_extractor(self):
        """
        Extracts information about gene regions from a genome.
        Returns:
            dict: A dictionary containing the extracted information about gene regions.
        """
        self.region_features = list(
            self.get_features_in_region(self.start + 1, self.end)
        )
        self.adapt_region_futures()
        return

    def adapt_region_futures(self):
        INDEX_OFFSET = 1

        if self.strand == "-":
            self.region_sequence = (
                self.fasta_dict[self.seqid]
                .seq[self.start : self.end]
                .reverse_complement()
            )
            seq_features = []

            for feature in self.region_features:
                new_start = self.end - feature.end
                new_end = self.end - feature.start + INDEX_OFFSET
                feature.start = new_start
                feature.end = new_end
                feature.strand = (
                    1  # Set the strand attribute to -1 when the strand is '-'
                )
                feature.type = feature.featuretype  # Preserve the type attribute
                feature = gffutilsFeatureToSeqFeature(feature).seqfeature
                feature.qualifiers["label"] = ["flanking_feature"]
                feature.qualifiers["label"] = ["flanking_feature"]
                seq_features.append(feature)
            self.region_features = seq_features

        if self.strand == "+":
            self.region_sequence = self.fasta_dict[self.seqid].seq[
                self.start : self.end
            ]
            seq_features = []

            for feature in self.region_features:
                feature.start -= self.start - INDEX_OFFSET
                feature.end -= self.start - INDEX_OFFSET
                feature.strand = (
                    1  # Set the strand attribute to -1 when the strand is '-'
                )
                feature.type = feature.featuretype  # Preserve the type attribute
                feature = gffutilsFeatureToSeqFeature(feature).seqfeature
                feature.qualifiers["label"] = ["flanking_feature"]
                seq_features.append(feature)
            self.region_features = seq_features
        return


class FlankingRegionConcatenator:
    """Only + strand and features transformed sequences please!
    Adapt all the features to the + strand"""

    def __init__(
        self,
        base_sequence,
        base_features,
        flanking_region_sequence,
        flanking_region_features,
    ):

        self.base_sequence = base_sequence
        self.base_features = base_features
        self.flanking_region_sequence = flanking_region_sequence
        self.flanking_region_features = [
            f for f in flanking_region_features if f.type == "CDS"
        ]

    def update_sequence_with_upstream_region(self):
        OFFSET = 1
        # Reindex the base_features to upstream sequence
        bp_upstream = len(self.flanking_region_sequence)
        updated_base_features = []
        for feature in self.base_features:
            update_feature = feature
            update_feature.location = FeatureLocation(
                feature.location.start + bp_upstream,
                feature.location.end + bp_upstream,
                strand=feature.location.strand,
            )
            updated_base_features.append(update_feature)
            self.updated_base_features = updated_base_features
        self.flanked_base_sequence = self.flanking_region_sequence + self.base_sequence
        self.flanking_and_base_features = (
            self.flanking_region_features + self.updated_base_features
        )

    def update_sequence_withdownstream_region(self):
        """TO DO"""
        return

    def update_sequence_with_upstream_and_downstream_region(self):
        """TO DO"""
        return


def gene_info_extractor(identifier: str, genome_db_path):
    """
    Extracts gene information from the genome database.

    Args:
        identifier (str): The identifier of the gene.
        genome_db_path: The path to the genome database.

    Returns:
        list: A list of gffutils.feature.Feature extracted from the gene.
    """

    gene_info = GeneInfoExtractor(genome_db_path)
    gene_id = identifier
    gene_info.gene = gene_id
    return gene_info


class GeneAnalyser:
    def __init__(
        self,
        db_path,
        fasta_path,
        gene,
        output_dir,
        up_stream_bp=None,
        down_stream_bp=None,
    ):
        """
        raw_info: GeneComponentExtractor that contains the inforamtion directly from the database.
        """
        self.db_path = db_path
        self.fasta_path = fasta_path
        self.db = gffutils.FeatureDB(self.db_path)
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(self.fasta_path, "fasta"))
        self.gene = gene
        self.output_dir = output_dir
        self.up_stream_bp = up_stream_bp
        self.down_stream_bp = down_stream_bp
        self.total_contig_length = len(self.fasta_dict[self.gene.seqid])
        self.raw_info = GeneComponentExtractor(self.db_path, self.fasta_path, self.gene)
        self.gene_indexes = None

    def run_me(self):
        # extract the raw sequence
        self.raw_sequence = SequenceExtractor(
            self.db_path,
            self.fasta_path,
            self.gene.seqid,
            self.gene.start,
            self.gene.end,
        )

        # update the raw info
        self.up_dated_info = RawInfoUpDater(
            self.total_contig_length, self.raw_sequence, self.raw_info
        )
        self.up_dated_info.run_me()
        if self.raw_info.gene.strand == "-":
            print("saving full sequence; strand is negative")

            # save the reverse complement sequence with introns
            make_genbank_file(
                self.gene.id,
                self.gene.id,
                "Reverse complement with adapted features",
                self.up_dated_info.updated_sequence,
                self.up_dated_info.features_gene_centric,
                self.raw_info.gene.featuretype,
                self.output_dir,
                "full_sequence",
            )
        else:
            # save the sequence with introns
            print("saving full sequence; strand is positive")
            make_genbank_file(
                self.gene.id,
                self.gene.id,
                "Sequence with adapted features",
                self.up_dated_info.updated_sequence,
                self.up_dated_info.features_gene_centric,
                self.raw_info.gene.featuretype,
                self.output_dir,
                "full_sequence",
            )
        print("Concatenating sequence and features")
        # concatenate the sequence and the features
        self.concatenated = Concatenater(
            self.up_dated_info.updated_sequence,
            self.up_dated_info.features_gene_centric,
        )
        # save the concatenated sequence and features
        make_genbank_file(
            self.gene.id,
            self.gene.id,
            "Concatenated sequence and features",
            self.concatenated.concatenated_sequence,
            self.concatenated.concatenated_features,
            self.raw_info.gene.featuretype,
            self.output_dir,
            "concatenated",
        )

        # extract the flanking region
        OFFSET = 1  # to start a base before or after the start base of the base gene
        print("Extracting flanking region")
        if self.raw_info.gene.strand == "-":
            start = self.raw_info.gene.end
            end = self.raw_info.gene.end + self.up_stream_bp
        if self.raw_info.gene.strand == "+":
            start = self.raw_info.gene.start - self.up_stream_bp
            end = self.raw_info.gene.start - OFFSET
        print("end, start", end, start)
        self.up_stream_flanking_region = GenomeRegionExtractor(
            self.db_path,
            self.fasta_path,
            start,
            end,
            self.raw_info.gene.seqid,
            self.raw_info.gene.strand,
        )
        self.up_stream_flanking_region.region_extractor()

    def get_upstream_sequence_for_concatenated_sequence(self):
        print("Adding upstream sequence to the concatenated sequence")

        self.concatenated_upstream_sequence = FlankingRegionConcatenator(
            self.concatenated.concatenated_sequence,
            self.concatenated.concatenated_features,
            self.up_stream_flanking_region.region_sequence,
            self.up_stream_flanking_region.region_features,
        )
        self.concatenated_upstream_sequence.update_sequence_with_upstream_region()

        make_genbank_file(
            self.gene.id,
            self.gene.id,
            f"Concatenated sequence and features with{self.up_stream_bp} bp upstream",
            self.concatenated_upstream_sequence.flanked_base_sequence,
            self.concatenated_upstream_sequence.flanking_and_base_features,
            self.raw_info.gene.featuretype,
            self.output_dir,
            "upstream_plus_concatenated",
        )

    def get_upstream_sequence_for_updated_sequence(self):
        print("Adding upstream sequence to the full sequence")
        self.upstream_sequence = FlankingRegionConcatenator(
            self.up_dated_info.updated_sequence,
            self.up_dated_info.features_gene_centric,
            self.up_stream_flanking_region.region_sequence,
            self.up_stream_flanking_region.region_features,
        )
        self.upstream_sequence.update_sequence_with_upstream_region()

        make_genbank_file(
            self.gene.id,
            self.gene.id,
            f"Sequence and features with {self.up_stream_bp} bp upstream",
            self.upstream_sequence.flanked_base_sequence,
            self.upstream_sequence.flanking_and_base_features,
            self.raw_info.gene.featuretype,
            self.output_dir,
            "upstream_plus_full_sequence",
        )


def genome_explorer(identifies: list, genome_db_path, genome_fasta_path, output):
    for gene_identifier in identifies:
        gene_list = gene_info_extractor(gene_identifier, genome_db_path)
        print(f"\033[92m{gene_list.gene}\033[0m")
        i = 0
        for gene in gene_list.gene:
            analyzer = GeneAnalyser(
                genome_db_path,
                genome_fasta_path,
                gene,
                output_dir=f"./{output}/{gene_identifier}_{i}/",
                up_stream_bp=5000,
            )
            # Run the analysis
            analyzer.run_me()
            analyzer.get_upstream_sequence_for_concatenated_sequence()
            analyzer.get_upstream_sequence_for_updated_sequence()
            i += 1
