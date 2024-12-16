import streamlit as st
from data_load import load_data
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

def create_chat_engine(base_index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=3500)
    llm = OpenAI(model="gpt-4", max_tokens=1000)
    
    return base_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=llm,
        streaming=True,
        verbose=True,
        system_prompt=(
            "You are an AI assistant analyzing Dan's personal documents. "
            "Before responding to any query:\n"
            "1. Thoroughly search ALL provided documents for relevant information\n"
            "2. If you find ANY relevant information, include it in your response\n"
            "3. Include direct quotes to support your answers\n"
            "4. If you're not completely sure about something, explain what you found "
            "and what might be missing"
        ),
        similarity_top_k=4,
        context_window=3000
    )

def chat_engine_generator(chat_engine, my_prompt):

    # Stream the response
    response = chat_engine.stream_chat(my_prompt)
    response_text = ""
    message_placeholder = st.empty()

    # Stream tokens
    for token in response.response_gen:
        response_text += str(token)
        message_placeholder.markdown(response_text + "▌")
    message_placeholder.markdown(response_text)

    # Show source documents if enabled
    if hasattr(response, 'source_nodes'):
        st.write("---")
        st.write("📚 Sources Used:")
        for node in response.source_nodes:
            with st.expander(f"Source: {node.metadata['filename']}"):
                st.write(f"Score: {node.score:.4f}") 

    return response_text

def main():
    st.title("My Big Brain")

    # Add configuration controls in sidebar
    st.sidebar.title("Chat Configuration")
    # Reset button
    if st.sidebar.button("Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Load the base index (cached)
    base_index = load_data()
    
    # Create chat engine with current settings
    chat_engine = create_chat_engine(base_index)

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
            response = chat_engine_generator(chat_engine, prompt)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()