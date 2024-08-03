### OpenAI Model.  Import the key
from dotenv import load_dotenv

load_dotenv()
import os

os.environ.get('OPENAI_API_KEY')

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

st.title("Your Subconscious")
st.markdown("On first load, wait for your data to ingest")

@st.cache_data
def load_data():
    # The below loads the data

    documents = SimpleDirectoryReader("data_7_7_24").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # This sets memory limits
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    llm = OpenAI(model="gpt-4")

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        system_prompt=(
            "You are the subconscious of Dan Stern, who has written all of his thoughts down about his life. \
                Use the files from his life to help him learn, improve and remember his life, specific to him.\
                    Do not use generic responses. Each file name represents a day he wrote something down \
                        or a specific idea or thing he wrote about.\
                         Everything needs to be specific to what he wrote.  You need to \
                        read between the lines in many cases.  Keep your responses casual, not formal."
        ),
    )
    return chat_engine

    

### This is where the magic is going to happen
def main():
    chat_engine = load_data()
    st.success("Success message")

    def chat_engine_generator(my_prompt):
        ## Chat Engine Generator
        response = chat_engine.chat(my_prompt)
        return response

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Tell Me?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("ai"):
            response = chat_engine_generator(prompt)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()