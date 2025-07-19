import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

# --- Load environment variables directly from .env ---
# This method works well when running locally or in Codespaces with a .env file.
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Still needed for embeddings

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set the GROQ_API_KEY environment variable in your .env file.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable for embeddings.")
    st.stop()


# --- Initialize Groq LLM and Google Generative AI Embeddings ---
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.4, # Keep temperature moderate for factual responses
    streaming=True,
    max_tokens=1024 # Added max_tokens for Groq to ensure longer outputs if needed
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


# --- UPDATED KNOWLEDGE BASE WITH YOUR RESUME CONTENT (CONFIRMED) ---
dummy_texts = [
    # Contact Information
    "Nithin Shetty M can be contacted via email at email adress: shettyn517@gmail.com and phone at +91 77601 52732.",
    "His LinkedIn profile is available at linkedin :  https://linkedin.com/in/nithin-shetty-m-530274265.",
    "His GitHub profile is https://github.com/nithinshettygit.",
    "Nithin is located in Dakshina Kannada, Karnataka, 574325.",

    # Executive Summary
    "Nithin Shetty M is an AI & ML engineering student with hands-on experience in deep learning, LLM-based agents, and real-time systems.",
    "He is skilled in deploying AI/ML solutions using LangChain, FAISS, and Python.",
    "Nithin is actively seeking AI/ML/GenAI Engineer roles to contribute to impactful, intelligent applications.",

    # Education
    "Nithin is a Final Year B.E. student in Artificial Intelligence and Machine Learning at Vivekananda College of Engineering and Technology (VCET), Puttur, Dakshina Kannada, Karnataka.",
    "He is affiliated with Visvesvaraya Technological University, Belagavi, Karnataka, and is expected to graduate in June 2026.",
    "Nithin maintains a strong academic record with a CGPA of 8.60 / 10.00 (First 6 Semesters) at VCET.",
    "He completed his Pre-University Course (PUC) with PCMB specialization at Sri Rama Pre-University College Kalladka, Dakshina Kannada, Karnataka, completing in 2022 with 88.00%.",
    "For his secondary education (SSLC), Nithin attended Shri Ramachandra High School Perne, Dakshina Kannada, Karnataka, completing in 2020 with a percentage of 96.36%.",

    # Experience & Projects
    # Project: AIRA – AI Powered Smart Teaching Robot
    "Project Title: AIRA – AI Powered Smart Teaching Robot.",
    "This was Nithin's Major Project at VCET, conducted from March 2025 to June 2025.",
    "Nithin built a RAG (Retrieval Augmented Generation) and LLM based AI teaching agent that features real-time Q&A with interruption-resume logic.",
    "He enhanced the response time and answer relevance by integrating FAISS vector search into the AIRA Teaching Robot.",
    "The tools used for the AIRA project include LangChain, GeminiFlash LLM (as part of the LLM integration), Sentence-Transformers, FAISS, Python, React.js, and FastAPI.",

    # Project: Autonomous Wheelchair using Deep Learning
    "Project Title: Autonomous Wheelchair using Deep Learning.",
    "This was a Mini Project at VCET, conducted from July 2024 to October 2024.",
    "Nithin developed a CNN-based Deep learning model for real-time direction control of a wheelchair prototype.",
    "He integrated the system with ESP8266 hardware and a Flask UI for dual-mode navigation.",
    "The project achieved reliable navigation in a controlled hospital-like environment.",
    "Tools used for the Autonomous Wheelchair project include Python, PyTorch, OpenCV, Flask, Arduino, and NodeMCU.",

    # Project: Hand Gesture Controlled Wheelchair (from previous content, assuming it's still relevant)
    "Project Title: Hand Gesture Controlled Wheelchair",
    "Description: This project demonstrates intuitive control of a wheelchair using real-time hand gesture recognition through computer vision.",
    "Nithin's Role: Nithin designed and implemented the computer vision pipeline for accurate gesture detection and mapping gestures to wheelchair movements.",
    "Technologies: Python, OpenCV, MediaPipe, Arduino (for motor control).",
    "Features: Allows users to control wheelchair direction (forward, backward, left, right, stop) with simple hand movements, offering an alternative control interface.",
    "Impact: Provides an accessible and user-friendly control mechanism, particularly beneficial for users who may have difficulty with traditional joysticks.",
    "GitHub Link for Hand Gesture Controlled Wheelchair: https://github.com/nithinshettygit/Hand-Gesture-Controlled-Wheelchair",

    # Technical Skills
    "Nithin's programming languages include Python and Java.",
    "His AI/ML skills cover PyTorch, scikit-learn, OpenCV, NLP (Natural Language Processing), and Generative AI.",
    "He is proficient with LLM & GenAI tools such as LangChain, OpenAI API, LLaMA, FAISS, RAG (Retrieval Augmented Generation), and Sentence-Transformers.",
    "For Web & UI development, Nithin has experience with Flask, Streamlit, and React.js.",

    # Soft Skills
    "Nithin possesses strong soft skills including Communication, Teamwork, and Problem-solving.",

    # Certifications
    "Nithin has a certification from Udemy: 'AI & LLM Engineering Mastery: GenAI, RAG Complete Guide'."
]


# --- Function to get vector store (cached) ---
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
        st.stop()

vectorstore = get_vectorstore(dummy_texts, embeddings)

# --- Define the custom prompt for the chatbot (MORE ROBUST) ---
CUSTOM_PROMPT_TEMPLATE = """You are Nithin Shetty M's highly detailed and helpful AI portfolio assistant.
Your primary goal is to provide comprehensive and accurate answers about Nithin based *only* on the provided context.
**Crucially, prioritize providing all available details from the context for every relevant question.**
If a question asks about Nithin's projects, ensure you mention *all* projects found in the context and provide their full descriptions, roles, technologies, features, and impact if available.
If a question is about Nithin's skills, list all relevant skills (programming languages, AI/ML, LLM/GenAI tools, Web/UI, soft skills).
**If asked about contact details, email, or phone, you MUST provide ALL available contact information including the full email address  (e.g., shettyn517@gmail.com), phone number, LinkedIn, and GitHub links.**
If the information is NOT present in the provided context, state clearly and politely: "I apologize, but I cannot find information on that specific topic within Nithin's provided portfolio context. I can answer questions about his skills, projects, education, experience, contact details, and certifications. For more details contact Nithin via email : shettyn517@gmail.com"
**Do NOT invent or infer any information.**

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


# --- Initialize conversation memory (Using ConversationBufferWindowMemory for efficiency) ---
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        k=5 # Keep only the last 5 turns of conversation for efficiency
    )

# --- Initialize ConversationalRetrievalChain ---
@st.cache_resource
def get_conversation_chain(_llm_model: ChatGroq, _vector_store: FAISS, _memory: ConversationBufferWindowMemory):
    if _vector_store is None:
        st.warning("Vector store is not available, conversation chain cannot be initialized.")
        return None
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_model,
            # Increased k to retrieve more documents, increasing chances of getting all relevant project details
            retriever=_vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=_memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=False # Set to True if you want to see what docs were retrieved
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        st.info("Please check if your LLM and embeddings models are correctly configured and accessible.")
        return None

conversation_chain = get_conversation_chain(llm, vectorstore, st.session_state.conversation_memory)


# --- Streamlit UI Components & Custom Design ---
# Changed layout to "wide" to gain more control with max-width in CSS, then control via CSS
st.set_page_config(page_title="Nithin's AI Assistant", page_icon="🤖", layout="wide") # Changed to wide

# --- Custom CSS for enhanced design and reduced size ---
st.markdown("""
<style>
/* Hide Streamlit header (including "Fork", "GitHub", "Deploy" if present) */
[data-testid="stToolbar"] {
    display: none; /* This hides the entire top-right bar */
    visibility: hidden; /* Added for belt-and-braces approach */
    height: 0; /* Ensures it takes no space */
}

/* Hide the main menu button (three dots) if it somehow remains */
#MainMenu {
    visibility: hidden;
    height: 0;
    overflow: hidden;
}

/* Hide the "Made with Streamlit" footer */
footer {
    visibility: hidden;
    height: 0;
}


/* General App Background and Layout */
.stApp {
    background-color: #f0f2f6; /* Light grey background for the entire app */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333;
}

/* Main content block background - REDUCED MAX-WIDTH */
.main .block-container {
    background-color: #ffffff; /* White background for the main content area */
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 1.5rem; /* Slightly reduced padding */
    margin-top: 1.5rem; /* Slightly reduced margin */
    margin-bottom: 1.5rem; /* Slightly reduced margin */
    max-width: 600px; /* IMPORTANT: Reduced max-width for a narrower app */
    margin-left: auto; /* Center the container */
    margin-right: auto; /* Center the container */
}

/* Chat message container styling */
div[data-testid="stChatMessage"] { /* General styling for all chat messages */
    padding: 8px 12px; /* Reduced padding */
    border-radius: 16px; /* Slightly smaller radius */
    margin-bottom: 8px; /* Reduced margin */
    max-width: 70%; /* Further limit message bubble width */
    font-size: 0.9rem; /* Slightly smaller font */
    line-height: 1.4; /* Slightly tighter line height */
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* User message bubble */
div[data-testid="stChatMessage"][data-st-chat-message-user="true"] {
    background-color: #e0f2f7 !important;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
    margin-left: auto;
    border: 1px solid #cceeff !important;
}
div[data-testid="stChatMessage"][data-st-chat-message-user="true"] p {
    color: #212121 !important;
    -webkit-text-fill-color: #212121 !important;
}

/* Assistant message bubble */
div[data-testid="stChatMessage"][data-st-chat-message-user="false"] {
    background-color: #f7f7f7 !important;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
    margin-right: auto;
    border: 1px solid #eaeaea !important;
}
div[data-testid="stChatMessage"][data-st-chat-message-user="false"] p {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Chat input bar styling */
.stTextInput > div > div > input {
    border-radius: 20px; /* Slightly smaller border radius */
    padding: 8px 15px; /* Reduced padding */
    border: 1px solid #ccc;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    font-size: 0.9rem; /* Slightly smaller font */
}

/* Send button styling */
.stButton button {
    background-color: #007bff;
    color: white;
    border-radius: 20px; /* Slightly smaller border radius */
    padding: 8px 15px; /* Reduced padding */
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
    font-size: 0.9rem; /* Slightly smaller font */
}

.stButton button:hover {
    background-color: #0056b3;
}

/* Spinner styling */
.stSpinner > div > div {
    color: #007bff;
}

/* Adjust title font */
h1 {
    color: #007bff;
    text-align: center;
    font-weight: 600;
    font-size: 1.8em; /* Slightly smaller title font */
    margin-bottom: 0.8em;
}

/* Styling for the new caption container */
.caption-container {
    background-color: #e6f7ff;
    border: 1px solid #cceeff;
    border-radius: 8px;
    padding: 0.6rem 1rem; /* Reduced padding */
    margin-top: 1rem; /* Reduced margin */
    margin-bottom: 1.5rem; /* Reduced margin */
    text-align: center;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    width: fit-content;
    max-width: 90%;
    margin-left: auto;
    margin-right: auto;
}

/* Styling for the caption text itself, within the container */
.caption-container p {
    color: #336699;
    font-size: 1.0em; /* Slightly smaller font size */
    margin: 0;
    line-height: 1.3;
}

/* Clear Chat button specific styling for smaller width */
.small-button-style {
    width: auto;
    padding: 6px 12px !important; /* Further reduced padding */
    font-size: 0.8rem !important; /* Further reduced font */
    border-radius: 18px !important; /* Slightly smaller border radius */
    background-color: #dc3545 !important;
    color: white !important;
    border: none !important;
}
.small-button-style:hover {
    background-color: #c82333 !important;
}

/* Info box styling */
.stAlert.info {
    background-color: #e6f7ff;
    border-left: 5px solid #2196F3;
    color: #2196F3;
    border-radius: 5px;
    padding: 10px;
}

/* Ensure messages fill container and are scrollable if needed - REDUCED MAX-HEIGHT */
.st-emotion-cache-zt5ig8 { /* This targets the container holding the chat messages */
    overflow-y: auto;
    max-height: 50vh; /* IMPORTANT: Reduced max-height for a more compact chat window */
    padding-right: 10px; /* Space for scrollbar */
}
</style>
""", unsafe_allow_html=True)


# Changed emoji to a robot head
st.title("🤖 Nithin's AI Portfolio Assistant")

# Use markdown to create a styled container for the caption
st.markdown(
    """
    <div class="caption-container">
        <p>Ask me anything about Nithin Shetty M's projects, skills, and experience!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT HANDLING (INCLUDING GREETINGS AND STREAMING) ---
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
        if re.fullmatch(r"\b" + greeting_pattern + r"\b.*", lower_prompt):
            is_greeting = True
            break

    # Placeholder for the bot's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if is_greeting:
            # Custom polite responses for greetings
            if "how are you" in lower_prompt or "how's it going" in lower_prompt:
                full_response = "I'm an AI, so I don't have feelings, but I'm ready to help! How can I assist you with Nithin's portfolio today?"
            elif "good morning" in lower_prompt:
                full_response = "Good morning! How can I help you learn about Nithin's work?"
            elif "good afternoon" in lower_prompt:
                full_response = "Good afternoon! What would you like to know about Nithin?"
            elif "good evening" in lower_prompt:
                full_response = "Good evening! I'm here to answer your questions about Nithin's portfolio."
            else:
                full_response = "Hello there! I'm Nithin's AI assistant. How can I help you explore his portfolio?"
            
            # Use inline style to force black text for greetings
            message_placeholder.markdown(f'<p style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; margin-bottom: 0;">{full_response}</p>', unsafe_allow_html=True)

        else:
            if conversation_chain:
                with st.spinner("Thinking..."):
                    try:
                        response_obj = conversation_chain.invoke(
                            {"question": prompt, "chat_history": st.session_state.conversation_memory.buffer_as_messages}
                        )
                        full_response = response_obj["answer"]
                        
                        # Use inline style to force black text for AI responses
                        message_placeholder.markdown(f'<p style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; margin-bottom: 0;">{full_response}</p>', unsafe_allow_html=True)

                    except Exception as e:
                        full_response = f"An error occurred while getting a response: {e}. Please try again."
                        st.error(full_response)
            else:
                full_response = "Chatbot is not fully initialized. Please check the backend configuration."
                # Use inline style to force black text for initialization message
                st.markdown(f'<p style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; margin-bottom: 0;">{full_response}</p>', unsafe_allow_html=True)

    # Add assistant message to chat history for future context
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Clear Chat Button (Moved to bottom and styled for smaller size) ---
st.markdown("---") # Add a separator for better visual grouping

# This button is explicitly styled with a custom class 'small-button-style' through JS below.
# Setting type="secondary" helps in targeting it with CSS/JS.
if st.button("Clear Chat", key="clear_chat_button", help="Clear all messages from the chat.", type="secondary"):
    st.session_state.messages = []
    st.session_state.conversation_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        k=5
    )
    st.rerun()

# Apply the custom class to the clear chat button using JavaScript
st.markdown(
    """
    <script>
        const clearButton = window.parent.document.querySelector('button[data-testid="stButton-secondary"]');
        if (clearButton) {
            clearButton.classList.add('small-button-style');
        }
    </script>
    """,
    unsafe_allow_html=True
)