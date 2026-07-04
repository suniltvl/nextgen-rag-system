from src.config import settings
from src.config.loader import load_config
from src.models import VectorStoreProvider
from src.rag_components.data_loaders import DataLoaderFactory
from src.rag_components.chunkers import ChunkingFactory
from src.utils.logger import Logger
from langchain_core.documents import Document
from src.utils.device import DeviceManager
from src.rag_components.embeddings.factory import EmbedderFactory

env_settings = None
config = None
logger = None

def load_env_settings():
    """
    Load environment settings from .env file and set global env_settings variable
    """
    global env_settings
    env_settings = settings
    
def hello():
    """
    Simple function to print a hello message to the console. 
    This is just for testing purposes and can be removed later.
    """
    logger.info("=" * 90)
    logger.info("Hello from real-world-rag-system!")
    logger.info("=" * 90)

def get_device():
    """
    Get the device to use for computation
    """
    return DeviceManager.get_device()


def main():

    doc_data = data_loader()
    chunker_instance = chunker()

    chunks = []
    ds_chunks_meta = []
    doc_id_counter = 0
    chunk_id_counter = 0

    if(len(doc_data) == 0):
        logger.warning("No documents found in the dataset. Exiting.")
        return
    
    if(chunker_instance is None):
        logger.warning("Chunker instance is None. Exiting.")
        return
    
    # chunks = chunker_instance.chunk(doc_data[0])
    # logger.debug(f"Number of chunks: {len(chunks)}, First chunk: {chunks[0]}")
    
    # return 
    for doc in doc_data:
        doc_chunks = chunker_instance.chunk(doc)
        chunks.extend(doc_chunks)
        doc_id_counter += 1
        for chunk in doc_chunks:
            chunk_id_counter += 1
            chunk_len = len(chunk)
            ds_chunks_meta.append(
                {
                    "document_id": doc_id_counter,
                    "chunk_id": chunk_id_counter, 
                    "chunk_text_length": chunk_len
                }            
            )


    logger.info(f"Number of chunks created: {len(chunks)}")
    # logger.info(f"Chunk metadata: {ds_chunks_meta}")    



def load_config_from_file():
    """
    Load configuration from file and set global config variable        
    """

    global config
    
    logger.info(f"Loading configuration from file... {settings.pipeline_config_file}")

    config = load_config(settings.pipeline_config_file)
    logger.debug(f"Data loader config: {config.data_loader}")

def data_loader():
    """
    Data loading component of the RAG system. This function initializes the data loader based on the configuration and loads the data.
    """
    
    logger.info("Data loader initialized")
    loader = DataLoaderFactory.create(config.data_loader)
    
    logger.debug(f"Loader created: {loader}")
    
    ds = loader.load() 
    logger.info(f"Data loaded: {len(ds)} documents found")
    return ds["documents"]

def chunker():
    """
    Chunking component of the RAG system. This function initializes the chunker based on the configuration and chunks the data.
    """
    logger.info("Chunker initialized")
    chunker = ChunkingFactory.create(config.chunking)
    return chunker
    
def embedder():
    """
    Embedding component of the RAG system. This function initializes the embedder based on the configuration and embeds the data.
    """
    logger.info("embedding initialized")
    embedder = EmbedderFactory.get_embedder(
        config.embedding.provider,
        config.embedding.model
    )

    print(embedder.embed_query("Hello world"))
    return embedder


if __name__ == "__main__":
    logger = Logger() # Initialize logger before loading environment settings        
    
    hello() # Print hello message to console
    load_env_settings() # Load environment settings after logger is initialized
    
    load_config_from_file()

    # device = get_device()
    # logger.info(f"Using device: {device}")

    embedder()
    # main() # Run the main function
