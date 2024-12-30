### OpenAI Model.  Import the key
### python3 -m streamlit run chat_data_functions.py
from dotenv import load_dotenv

load_dotenv()
import os
import time

## Call the functions

import streamlit as st
from pathlib import Path

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

# os.environ.get('OPENAI_API_KEY')
# os.environ.get('LLAMA_CLOUD_API_KEY')

openai_api_key = st.secrets["OPENAI_API_KEY"]
llama_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]

st.title("Your Subconscious")
st.markdown("On first load, wait for your data to ingest")

DEFAULT_DATA_PATH = "data_7_7_24"

if 'data_path' not in st.session_state:
    st.session_state['data_path'] = DEFAULT_DATA_PATH

@st.cache_resource
def load_data():
    # The below loads the data
    with st.spinner("Loading documents"):
        start_time = time.time()
    
    if not Path(st.session_state.data_path).exists():
        st.error(f"❌ Path {st.session_state.data_path} does not exist!")
        return None

    parser = LlamaParse(result_type="markdown")
    documents = SimpleDirectoryReader(
        st.session_state.data_path, 
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
