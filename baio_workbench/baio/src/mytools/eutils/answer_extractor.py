import json
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class EutilsAnswerExtractor:
    """Extract answer for eutils and blast results"""

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        template_api_eutils = """
        You have to answer the question:{question} as clear and short as possible manner
        , be factual!\n\
        Example question: What is the official gene symbol of LMP10?
        Output to find answer in: [b'1. Psmb10 Official Symbol: Psmb10 and Name:
        proteasome (prosome, macropain) subunit, beta type 10 [Mus musculus (house
        mouse)] Other Aliases: Mecl-1, Mecl1 Other Designations: proteasome subunit
        beta type-10; low molecular mass protein 10; macropain subunit MECl-1;
        multicatalytic endopeptidase complex subunit MECl-1; prosome Mecl1; proteasome
        (prosomome, macropain) subunit, beta type 10; proteasome MECl-1; proteasome
        subunit MECL1; proteasome subunit beta-2i\nChromosome: 8; Location: 8 53.06 cM
        \nAnnotation: Chromosome 8 NC_000074.7 (106662360..106665024, complement)\n
        ID: 19171\n\n2. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit
        beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10, MECL1, PRAAS5, beta2i\n
        Other Designations: proteasome subunit beta type-10; low molecular mass protein
        10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit
        MECl-1; proteasome (prosome, macropain) subunit, beta type, 10; proteasome
        MECl-1; proteasome catalytic subunit 2i; proteasome subunit MECL1; proteasome
        subunit beta 10; proteasome subunit beta 7i; proteasome subunit beta-2i;
        proteasome subunit beta2i Chromosome: 16; Location: 16q22.1 Annotation:
        Chromosome 16 NC_000016.10 (67934506..67936850, complement) MIM: 176847 ID:
        5699  3. MECL1 Proteosome subunit MECL1 [Homo sapiens (human)] Other Aliases:
        LMP10, PSMB10 This record was replaced with GeneID: 5699 ID: 8138  ']\
        Answer: PSMB10\n\
        Example question: Which gene is SNP rs1217074595 associated with?
        Output to find answer in: [b   header :  type : esummary , version : 0.3  ,
        result :  uids :[ 1217074595 ], 1217074595 :  uid : 1217074595 , snp_id
        :1217074595, allele_origin :  , global_mafs :[  study : GnomAD , freq :
        A=0.000007/1  ,  study : TOPMED , freq : A=0.000004/1  ,  study : ALFA , freq :
        A=0./0  ], global_population :  , global_samplesize :  , suspected :  ,
        clinical_significance :  , genes :[  name : LINC01270 , gene_id : 284751  ],
        acc : NC_000020.11 , chr : 20 , handle : GNOMAD,TOPMED , spdi : NC_000020.11:
        50298394:G:A , fxn_class : non_coding_transcript_variant , validated :
        by-frequency,by-alfa,by-cluster , docsum : HGVS=NC_000020.11:g.50298395G>A,
        NC_000020.10:g.48914932G>A,NR_034124.1:n.351G>A,NM_001025463.1:c.*4G>A|SEQ=
        [G/A]|LEN=1|GENE=LINC01270:284751 , tax_id :9606, orig_build :155, upd_build
        :156, createdate : 2017/11/09 09:55 , updatedate : 2022/10/13 17:11 , ss :
        4354715686,5091242333 , allele : R , snp_class : snv , chrpos : 20:50298395 ,
        chrpos_prev_assm : 20:48914932 , text :  , snp_id_sort : 1217074595 ,
        clinical_sort : 0 , cited_sort :  , chrpos_sort : 0050298395 , merged_sort : 0
        \n ]\n\
        Answer: LINC01270\n\
        Example question: What are genes related to Meesmann corneal dystrophy?\n\
        Output to find answer in: [b   header :  type : esummary , version : 0.3  ,
        result :  uids :[ 618767 , 601687 , 300778 , 148043 , 122100 ], 618767 :  uid :
        618767 , oid : #618767 , title : CORNEAL DYSTROPHY, MEESMANN, 2; MECD2 ,
        alttitles :  , locus : 12q13.13  , 601687 :  uid : 601687 , oid : *601687 ,
        title : KERATIN 12, TYPE I; KRT12 , alttitles :  , locus : 17q21.2  , 300778 :
        uid : 300778 , oid : %300778 , title : CORNEAL DYSTROPHY, LISCH EPITHELIAL;
        LECD , alttitles :  , locus : Xp22.3  , 148043 :  uid : 148043 , oid : *148043 ,
        title : KERATIN 3, TYPE II; KRT3 , alttitles :  , locus : 12q13.13  , 122100 :
        uid : 122100 , oid : #122100 , title : CORNEAL DYSTROPHY, MEESMANN, 1; MECD1 ,
        alttitles :  , locus : 17q21.2    \n ]\
        Answer: KRT12, KRT3\
        Example question:
        Output to find answer in: [b'\n1. PSMB10\nOfficial Symbol: PSMB10 and Name:
        proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10,
        MECL1, PRAAS5, beta2i\nOther Designations:
        Answer: PSMB10
        Note: always format the answer nicely if you can.
        Base you answer on the information given here:\n\
        {context}
        """
        self.eutils_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=template_api_eutils
        )
        self.eutils_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=template_api_eutils
        )

    def query(self, question: str, file_path: str, llm, embedding) -> dict:
        # Read the file
        with open(file_path, "r") as file:
            content = file.read()

        # Create a temporary file and write the content to it
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        if os.path.exists(temp_file_path):
            loader = TextLoader(temp_file_path)
        else:
            print(f"Temporary file not found: {temp_file_path}")
            return {"answer": "Error: File not found"}

        documents = loader.load()
        os.remove(temp_file_path)

        # split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # embed
        doc_embeddings = FAISS.from_documents(docs, embedding)
        ncbi_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.eutils_CHAIN_PROMPT},
            verbose=True,
        )
        relevant_api_call_info = ncbi_qa_chain(question)
        return relevant_api_call_info


def result_file_extractor(
    question: str, uuid: str, log_file_path: str, llm, embedding
) -> dict:
    """Extracting the answer result file"""
    print("In result file extractor")
    try:
        # Read the log file
        with open(log_file_path, "r") as log_file:
            log_data = json.load(log_file)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {log_file_path}")
        return {"answer": "Error: Could not read log file"}
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
        return {"answer": "Error: Log file not found"}
    print(f"Log file path: {log_file_path}")
    print(f"UUID: {uuid}")

    answer_extractor = EutilsAnswerExtractor()

    for entry in log_data:
        if entry["uuid"] == uuid:
            file_path = entry.get("file_path")
            if not file_path:
                return {"answer": f"Error: No file found for UUID {uuid}"}
            try:
                answer = answer_extractor.query(question, file_path, llm, embedding)[
                    "answer"
                ]
            except Exception as e:
                print(f"Error extracting answer: {str(e)}")
                return {"answer": f"Error extracting answer: {str(e)}"}
            # Add the answer to the entry
            entry["answer"] = answer
            # Write the updated log data back to the file
            try:
                with open(log_file_path, "w") as log_file:
                    json.dump(log_data, log_file, indent=2)
            except Exception as e:
                print(f"Error writing updated log file: {str(e)}")
                return {"answer": answer, "warning": "Answer not saved to log file"}
            return {"answer": answer}
    # If we didn't find a matching UUID
    return {"answer": f"Error: No entry found for UUID {uuid}"}
