## Introduction

Welcome to BaIO, a Streamlit-based tool designed to facilitate biological data interaction. BaIO seamlessly connects you to multiple biological databases and analysis tools using natural language queries, making complex data retrieval and analysis more accessible.

![Screenshot of Baio](baio_overview.png)

### Key Features

1. Natural Language Queries: Ask questions in plain English about genes, proteins, or biological processes.
2. Multi-Database Integration through 'Agents':

| Database Name | Website | Agent | What | Key Features | Use Cases |
|---------------|---------|-------|------|--------------|-----------|
| ANISEED | [ANISEED](https://www.aniseed.cnrs.fr/) | Aniseed agent | Developmental biology data | Ascidian-specific |Embryo development studies |
| NCBI BLAST | [NCBI](https://www.ncbi.nlm.nih.gov/) | BaIO agent | Sequence search | BLAST algorithm| • Sequence homology search<br>• Gene/protein identification |
| NCBI dbSNP | [NCBI](https://www.ncbi.nlm.nih.gov/) | BaIO agent | SNP database | SNP cataloging| Variant analysis|
| NCBI OMIM | [NCBI](https://www.ncbi.nlm.nih.gov/) | BaIO agent | Genetic disorders database | Disease-gene associations | Genetic disorder research |
| UCSC Genome browser | [UCSC Genome browser](https://genome.ucsc.edu/) | BaIO agent | Genome alignment | Genome mapping | ID genome loci for target sequence|
| Ensembl | [Ensembl](https://www.ensembl.org/) | BaIO agent | Genomic annotations | • Comparative genomics<br>• Regulatory features | Gene annotation |
| UniProt | [UniProt](https://www.uniprot.org/) | BaIO agent| Protein annotations | • Curated protein data<br>• Functional annotations | Protein function analysis |



3. Local File Analysis: Explore and annotate your own CSV files with biological data.

4. Interactive Results: View results in user-friendly formats, including tables and downloadable files.

### Use Cases

- Quickly find gene expression data across developmental stages for Ascidians
- Quick answer to 'What is this sequence'?
- Quick Gene symbol conversions
- Annotate a list of genes with GO terms and other identifiers
- Perform cross-database queries without needing to learn multiple query languages
