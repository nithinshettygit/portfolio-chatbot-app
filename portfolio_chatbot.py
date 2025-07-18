import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# --- Initialize Google Generative AI components ---
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

llm = get_llm()
embeddings = get_embeddings()

# --- Placeholder for your knowledge base (replace with your actual data loading) ---
# In a real scenario, you'd load your portfolio documents, chunk them, and create/load a FAISS vector store.
# For deployment, this part needs to be functional.
# Example: Let's assume you have some pre-processed texts or a way to generate a simple vector store.
# For simplicity, we'll create a dummy one. You should replace this with your actual portfolio content.

# Dummy text for demonstration if you don't have actual docs ready for deployment
dummy_texts = [
    "Nithin Shetty M is an AI/ML Engineering student at VCET Puttur. He is proficient in Python and Java.",
    "Nithin's projects include AIRA Teaching Bot, an Autonomous Wheelchair using Deep Learning, and a Hand Gesture Controlled Wheelchair.",
    "His soft skills include communication, teamwork, and problem-solving.",
    "You can contact Nithin via email at shettyn517@gmail.com or connect on LinkedIn.",
    "The AIRA Teaching Bot leverages AI to enhance learning experiences."
]

@st.cache_resource
def get_vectorstore(texts, embeddings_model):
    if not texts: # Ensure texts are not empty
        return None
    try:
        # Create a FAISS vector store from the dummy texts
        vectorstore = FAISS.from_texts(texts, embedding=embeddings_model)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

vectorstore = get_vectorstore(dummy_texts, embeddings)

if vectorstore is None:
    st.error("Could not initialize vector store. Chatbot functionality will be limited.")
    # Exit or handle gracefully if vectorstore creation fails
    st.stop() # Stop the app if crucial component fails


# --- Chatbot components ---
# Custom prompt to guide the chatbot
CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant for Nithin Shetty M's portfolio.
Answer the user's questions truthfully and concisely based ONLY on the provided context.
If you don't know the answer, say "I apologize, but that information isn't directly available in Nithin's portfolio. You can reach out to him directly via the contact section!" Do NOT make up answers.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=CUSTOM_PROMPT_TEMPLATE,
)

# Initialize conversation memory
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Important for ConversationalRetrievalChain
    )

# Initialize ConversationalRetrievalChain
@st.cache_resource
def get_conversation_chain(llm_model, vector_store, memory):
    if vector_store is None:
        return None
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_model,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=False # Set to True if you want to show sources
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        return None

conversation_chain = get_conversation_chain(llm, vectorstore, st.session_state.conversation_memory)

# --- Streamlit UI ---
st.set_page_config(page_title="Nithin's AI Assistant", page_icon="🤖")

st.title("💬 Nithin's AI Portfolio Assistant")
st.caption("Ask me anything about Nithin Shetty M's projects, skills, and experience!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Nithin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if conversation_chain:
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({"question": prompt})
            bot_response = response.get("answer", "I encountered an issue getting an answer.")
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Chatbot is not fully initialized. Please check configuration.")
        st.session_state.messages.append({"role": "assistant", "content": "Chatbot is not fully initialized. Please check configuration."})

# Disclaimer for dummy data
st.markdown("---")
st.info("Note: For demonstration, the knowledge base is a small set of dummy texts. A real-world application would use comprehensive portfolio documents for more detailed answers.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    st.experimental_rerun()