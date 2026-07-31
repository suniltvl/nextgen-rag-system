
from pathlib import Path



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
        "emanual": "TV manual", 
        "techqa":"Technotes"
    },
    'gk':
    {
        "hotpotqa":"wiki 1",
        "msmacro":"web pages",
        "hagrid":"wiki 2",
        "expertqa":"googlesearch"
    }
}

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