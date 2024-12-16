### OpenAI Model.  Import the key
### python3 -m streamlit run chat_data_functions.py
from dotenv import load_dotenv

load_dotenv()
import os
import time

os.environ.get('OPENAI_API_KEY')
os.environ.get('LLAMA_CLOUD_API_KEY')

## Call the functions

import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse

st.title("Your Subconscious")
st.markdown("On first load, wait for your data to ingest")

@st.cache_resource
def load_data():
    # The below loads the data
    with st.spinner("Loading documents"):
        start_time = time.time()

    parser = LlamaParse(result_type="markdown")
    documents = SimpleDirectoryReader(
        "data_7_7_24", 
        file_extractor=parser,
    ).load_data()

    for doc in documents:
        doc.metadata = {
            "filename": doc.metadata.get('file_name', ''),
            "text_preview": doc.text[:200]
        }
    
# Configure more granular text splitting
    from llama_index.core.node_parser import SentenceSplitter
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=100,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=parser,
    )


    elapsed_time = time.time() - start_time
    st.success(f'✅ Your Documents loaded in {elapsed_time:.2f} seconds!')

    return index
