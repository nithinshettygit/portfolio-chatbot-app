import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

# Load environment variables (for API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file or Streamlit secrets.")
    st.stop()

# --- Initialize Google Generative AI components in global scope ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


# --- UPDATED: Placeholder for your knowledge base with more content and links ---
# This is where you significantly expand your data.
dummy_texts = [
    # General Information
    "Nithin Shetty M is a dedicated AI/ML Engineering student currently pursuing his B.E. at Vivekananda College of Engineering & Technology (VCET) Puttur.",
    "He is passionate about developing intelligent systems and has a strong foundation in machine learning, deep learning, and data science.",
    "Nithin is actively seeking opportunities in AI/ML Engineering, Data Science, or Software Development roles.",

    # Skills
    "Nithin is highly proficient in programming languages such as Python and Java.",
    "His technical skills include Machine Learning, Deep Learning, Natural Language Processing (NLP), Computer Vision, and Data Analysis.",
    "He is experienced with frameworks and libraries like TensorFlow, Keras, PyTorch, Scikit-learn, Pandas, NumPy, and Streamlit.",
    "Nithin also has experience with version control using Git and GitHub.",

    # Projects
    "One of Nithin's key projects is 'AIRA Teaching Bot', an AI-powered educational assistant designed to enhance learning experiences.",
    "He developed an 'Autonomous Wheelchair' utilizing Deep Learning techniques for navigation and obstacle avoidance.",
    "Another innovative project is a 'Hand Gesture Controlled Wheelchair', demonstrating his skills in computer vision and embedded systems.",
    "Nithin's projects showcase his ability to apply AI/ML concepts to solve real-world problems.",
    "He has also worked on data analysis and visualization tasks, transforming raw data into actionable insights.",

    # Contact Information and Links (Crucial for adding value)
    "You can easily connect with Nithin Shetty M through his professional profiles.",
    "Nithin's GitHub profile is an excellent place to view his code, projects, and contributions. His GitHub link is: https://github.com/nithinshettygit",
    "For professional networking and his complete career history, you can find Nithin on LinkedIn. His LinkedIn profile link is: https://www.linkedin.com/in/nithin-shetty-m-8646b3226/",
    "If you wish to contact Nithin via email for inquiries or opportunities, his email address is: shettyn517@gmail.com",
    "Feel free to reach out to Nithin through his social media or professional channels.",

    # Education
    "Nithin is expected to graduate with a Bachelor of Engineering in Artificial Intelligence and Machine Learning from VCET Puttur.",
    "His coursework includes advanced topics in AI, ML algorithms, data structures, and software engineering principles."
]


@st.cache_resource
def get_vectorstore(texts: list[str], _embeddings_model: GoogleGenerativeAIEmbeddings):
    if not texts:
        st.warning("No texts provided for vector store creation.")
        return None
    try:
        vectorstore = FAISS.from_texts(texts, embedding=_embeddings_model)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.info("Ensure your GOOGLE_API_KEY is correct and has access to embedding models.")
        return None

vectorstore = get_vectorstore(dummy_texts, embeddings)

if vectorstore is None:
    st.error("Vector store could not be initialized. Chatbot functionality will be limited.")
    st.stop()


# --- Define the custom prompt for the chatbot ---
CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant for Nithin Shetty M's portfolio.
Answer the user's questions truthfully and concisely based ONLY on the provided context.
If the information is not directly in the context, politely state that you cannot answer from the available information.
Do NOT make up answers. Prioritize information directly related to Nithin's projects, skills, and experience.

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

# --- Initialize conversation memory ---
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

# --- Initialize ConversationalRetrievalChain ---
@st.cache_resource
def get_conversation_chain(_llm_model: ChatGoogleGenerativeAI, _vector_store: FAISS, _memory: ConversationBufferMemory):
    if _vector_store is None:
        return None
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_model,
            retriever=_vector_store.as_retriever(),
            memory=_memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=False
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        return None

conversation_chain = get_conversation_chain(llm, vectorstore, st.session_state.conversation_memory)


# --- Streamlit UI Components ---
st.set_page_config(page_title="Nithin's AI Assistant", page_icon="🤖")

st.title("💬 Nithin's AI Portfolio Assistant")
st.caption("Ask me anything about Nithin Shetty M's projects, skills, and experience!")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT HANDLING (INCLUDING GREETINGS) ---
if prompt := st.chat_input("Ask me about Nithin..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert prompt to lowercase for easier matching
    lower_prompt = prompt.lower().strip()

    # Define common greetings
    greetings = [
        r"hi", r"hello", r"hey", r"how are you", r"how's it going",
        r"good morning", r"good afternoon", r"good evening", r"what's up",
        r"namaste", r"heyy", r"hi there"
    ]

    # Check if the prompt is a greeting
    is_greeting = False
    for greeting_pattern in greetings:
        # Use re.fullmatch to ensure the entire input matches a greeting pattern
        if re.fullmatch(r"\b" + greeting_pattern + r"\b.*", lower_prompt):
            is_greeting = True
            break


    bot_response = ""
    if is_greeting:
        # Custom polite responses for greetings
        if "how are you" in lower_prompt or "how's it going" in lower_prompt:
            bot_response = "I'm an AI, so I don't have feelings, but I'm ready to help! How can I assist you with Nithin's portfolio today?"
        elif "good morning" in lower_prompt:
            bot_response = "Good morning! How can I help you learn about Nithin's work?"
        elif "good afternoon" in lower_prompt:
            bot_response = "Good afternoon! What would you like to know about Nithin?"
        elif "good evening" in lower_prompt:
            bot_response = "Good evening! I'm here to answer your questions about Nithin's portfolio."
        else: # General greetings like hi, hello, hey
            bot_response = "Hello there! I'm Nithin's AI assistant. How can I help you explore his portfolio?"
    else:
        # If not a greeting, proceed with the RAG chain
        if conversation_chain:
            with st.spinner("Thinking..."):
                try:
                    response = conversation_chain.invoke({"question": prompt})
                    bot_response = response.get("answer", "I apologize, but I could not process your request at this moment.")
                except Exception as e:
                    bot_response = f"An error occurred while getting a response: {e}. Please try again."
                    st.error(bot_response)
        else:
            bot_response = "Chatbot is not fully initialized. Please check the backend configuration."

    with st.chat_message("assistant"):
        # Use st.markdown to render links correctly
        st.markdown(bot_response)
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})


# Disclaimer for dummy data (useful for deployment demo)
st.markdown("---")
st.info("Note: This chatbot provides information based on Nithin's portfolio data. For comprehensive details, please refer to his full portfolio sections or direct contact information.")

# Button to clear the chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    st.rerun()