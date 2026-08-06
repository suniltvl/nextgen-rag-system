
from pathlib import Path
from datasets import load_dataset
from langchain_core.documents import Document

from langchain_chroma import Chroma
import os

from dotenv import load_dotenv
load_dotenv()



# Constants

DATASET_SOURCE = 'suniltvl/ragbench'
DATA_SPLIT = 'test'
VECTOR_DATABASES = ['chroma'] # ['chroma', 'milvus']
EMBEDDING_MODELS = ["BAAI/LLM-Embedder", "BAAI/bge-large-en-v1.5"]
MAX_CHUNKS = 5000
CHUNKING_SIZES = [256, 512, 1024]
CHUNKING_OVERLAPS = [50, 100, 200]
SEPARATORS = ["\n\n", "\n", " ", ".", ","]
DOMAINS = {
    'cs':
    {
        "delucionqa":"Jeep manual", 
        # "emanual": "TV manual", 
        # "techqa":"Technotes"
    },
    'gk':
    {
        "hotpotqa":"wiki 1",
        # "msmarco":"web pages",
        # "hagrid":"wiki 2",
        # "expertqa":"googlesearch"
    }
}

DOMAIN_NAME_MAPPING = {
    "cs": "Customer Support",
    "gk": "General Knowledge",
}

DATABASE_URL = os.getenv("LOCAL_POSTGRE_DATABASE_URL") #"postgresql://neondb_owner:npg_ev8q6OUgwtrn@ep-shiny-unit-ad1du3mx.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

RETRIEVAL_TYPE = "dense"
TOP_K = 5

LLM_PROVIDER = "openrouter"
LLM_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
GEN_LLM_MODEL = "nvidia/nemotron-nano-9b-v2:free" #"openai/gpt-4o-mini"
EVAL_LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"









def get_db_folder(db_type: str) -> Path:
    """
    Get the database folder path for a given database type.
    
    Args:
        db_type (str): The type of database (e.g., 'chroma', 'milvus')
    
    Returns:
        Path: The path to the database folder
    """
    return Path(f"../database_{db_type}")


def get_collection_name(
    embedding_model: str,
    domain_name_key: str,
) -> str:
    """
    Get the collection name for a given embedding model and domain name key.
    
    Args:
        embedding_model (str): The embedding model name
        domain_name_key (str): The domain name key
    
    Returns:
        str: The collection name
    """
    return f"{embedding_model.replace('/', '_')}-{domain_name_key}"


def get_persist_directory(
    db_type: str,
    domain_name_key: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    """
    Get the persist directory path for a given database type, domain name key, chunk size, and chunk overlap.
    
    Args:
        db_type (str): The type of database (e.g., 'chroma', 'milvus')
        domain_name_key (str): The domain name key
        chunk_size (int): The chunk size
        chunk_overlap (int): The chunk overlap
    
    Returns:
        Path: The path to the persist directory
    """
    return (
        get_db_folder(db_type)
        / f"{domain_name_key}_{chunk_size}_{chunk_overlap}"
    )



VECTOR_DB_CACHE = {}

def get_vector_db(persist_directory, collection_name, embedding_function):
    """
    Get a vector database instance for a given persist directory, collection name, and embedding function.
    
    Args:
        persist_directory (str): The persist directory path
        collection_name (str): The collection name
        embedding_function (Embeddings): The embedding function
    
    Returns:
        Chroma: The vector database instance
    """
    key = (persist_directory, collection_name)

    if key not in VECTOR_DB_CACHE:
        VECTOR_DB_CACHE[key] = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_function
        )

    return VECTOR_DB_CACHE[key]

def get_questions_table_name(param_domains):
    """
    Get the questions table name.
    
    Returns:
        str: The questions table name
    """
    questions_table = "qtn"

    domains_name = ""
    for domain in param_domains.keys():
        domains_name += f"_{domain}"

    questions_table += domains_name
    
    return questions_table


def get_eval_table_name(param_domains):
    """
    Get the evaluation table name.
    
    Returns:
        str: The evaluation table name
    """
    eval_table = "eval"

    domains_name = ""
    for domain in param_domains.keys():
        domains_name += f"_{domain}"

    eval_table += domains_name
    
    return eval_table


def get_docs(db_name: str):
    dataset = load_dataset(DATASET_SOURCE, db_name, split=DATA_SPLIT)
    dedup = deduplicate_data(dataset, db_name)
    docs = [
        Document(
            metadata=metadata, 
            page_content=content
        )
        for content, metadata in dedup.items()
    ]
    print(f"Loaded {len(docs)} documents from {db_name}")
    return docs

def deduplicate_data(data, doc_type):
    data_dict = {}
    for d in data:
        document = " ".join(d["documents"])
        if document in data_dict:
            data_dict[document]["docid"].append(d["id"])
        else:
            data_dict[document] = {"docid":[d["id"]]}
            
    for k in data_dict:
        data_dict[k]["document_type"] = doc_type
        
    return data_dict

def deduplicate_data_new(data, doc_type):
    data_dict = {}
    for d in data:
        if hasattr(d, "page_content"):
            document = d.page_content
            doc_id = d.metadata.get("docid", [None])[0] if getattr(d, "metadata", None) else None
        else:
            document = " ".join(d["documents"])
            doc_id = d["id"]
        if document in data_dict:
            if doc_id is not None:
                data_dict[document]["docid"].append(doc_id)
        else:
            data_dict[document] = {"docid": [doc_id] if doc_id is not None else []}
            
    for k in data_dict:
        data_dict[k]["document_type"] = doc_type
        
    return data_dict