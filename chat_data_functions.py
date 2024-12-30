import streamlit as st
from data_load import load_data, DEFAULT_DATA_PATH
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor
from pathlib import Path


def create_chat_engine(base_index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    llm = OpenAI(model="gpt-4o", max_tokens=1500)
    
    return base_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=llm,
        streaming=True,
        verbose=True,
        system_prompt=(
            "You are a knowledgeable assistant with access to the user's personal notes and memory. "
            "IMPORTANT: Always check and reference the memory context provided. Use this information to: "
            "1. Personalize your responses "
            "2. Reference past conversations and known facts "
            "3. Make connections between new information and what you remember "
            "4. Correct any outdated information you find "
                "Core Guidelines:\n"
                "1. Always perform fresh semantic searches for each question, even in ongoing conversations\n"
                "2. Look for connections between notes that might not be immediately obvious\n" 
                "3. When answering follow-up questions, don't just rely on the previous context - actively search for additional relevant notes\n"
                "4. If the current context seems insufficient, explicitly mention other notes that might be worth exploring\n"
                "5. When referencing notes, ALWAYS use the exact format: [[filename]] - double brackets with no spaces\n"
                "6. Be concise but thorough in your responses\n\n"

                "When referencing notes:\n"
                "- Use the exact format: [[filename.md]]\n"
                "- Never use single brackets, single parentheses, or double parentheses\n"
                "- Always include the .md extension if it isn't already present\n"
                "- Never add spaces between brackets and filename\n\n"

                "When synthesizing information:\n"
                "- Clearly distinguish between information from notes and general knowledge\n"
                "- Point out interesting connections between different notes\n"
                "- If you notice gaps in the available information, suggest areas where the user might want to add more notes\n"
                "- When appropriate, encourage the user to explore related topics in their notes\n\n"

                "Remember: Each new question is an opportunity to discover new connections in the users notes, even if it seems related to the previous conversation."
            # "You are an AI assistant analyzing Dan's personal documents. "
            # "Before responding to any query:\n"
            # "1. Thoroughly search ALL provided documents for relevant information\n"
            # "2. If you find ANY relevant information, include it in your response\n"
            # "3. Include direct quotes to support your answers\n"
            # "4. If you're not completely sure about something, explain what you found "
            # "and what might be missing"
            # "5. The way the documents are structured is that there might be an empty document, but the"
            # "title of that document can be found and included in other documents as a subject.  Please reference those still."
        ),
        similarity_top_k=20,
        context_window=5000,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
        context_builder_kwargs={
            "should_trim_context": True
        }
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
    with st.sidebar:
        st.sidebar.title("Chat Configuration")

        # Initialize data path in session state if not present
        if 'data_path' not in st.session_state:
            st.session_state.data_path = DEFAULT_DATA_PATH
        
        # Show current path
        st.write("📂 Current Data Path:")
        st.code(st.session_state.data_path)
        
        # Path input and button in a form
        with st.form("path_form"):
            new_path = st.text_input("Enter new data path:", key="new_path_input")
            submit = st.form_submit_button("Load New Path")
            
            if submit and new_path:
                if Path(new_path).exists():
                    st.session_state.data_path = new_path
                    st.cache_resource.clear()
                    st.experimental_rerun()
                else:
                    st.error("❌ Path does not exist!")

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