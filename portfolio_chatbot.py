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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Still needed for embeddings

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set the GROQ_API_KEY environment variable in your .env file or Streamlit secrets.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable for embeddings.")
    st.stop()


# --- Initialize Groq LLM and Google Generative AI Embeddings ---
# Increased max_output_tokens slightly for potentially longer answers
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.4, # Keep temperature moderate for factual responses
    streaming=True,
    max_tokens=1024 # Added max_tokens for Groq to ensure longer outputs if needed
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


# --- UPDATED KNOWLEDGE BASE WITH YOUR RESUME CONTENT (CONFIRMED) ---
# This content is taken directly from your provided resume text.
dummy_texts = [
    # Contact Information
    "Nithin Shetty M can be contacted via email at shettyn517@gmail.com or phone at +91 77601 52732.",
    "His LinkedIn profile is available at https://linkedin.com/in/nithin-shetty-m-530274265.",
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
Always include full links (GitHub, LinkedIn, email) if they are mentioned and relevant to the question.
If the information is NOT present in the provided context, state clearly and politely: "I apologize, but I cannot find information on that specific topic within Nithin's provided portfolio context. I can answer questions about his skills, projects, education, experience, contact details, and certifications."
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
        return None

conversation_chain = get_conversation_chain(llm, vectorstore, st.session_state.conversation_memory)


# --- Streamlit UI Components & Custom Design ---
st.set_page_config(page_title="Nithin's AI Assistant", page_icon="🤖", layout="centered")

# --- Custom CSS for enhanced design ---
st.markdown("""
<style>
/* General App Background and Layout */
.stApp {
    background-color: #f0f2f6; /* Light grey background for the entire app */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333;
}

/* Main content block background */
.main .block-container {
    background-color: #ffffff; /* White background for the main content area */
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 2rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Chat message container styling */
.stChatMessage {
    padding: 10px 15px;
    border-radius: 18px;
    margin-bottom: 10px;
    max-width: 75%; /* Limit message bubble width */
    font-size: 0.95rem;
    line-height: 1.5;
}

/* User message bubble */
.stChatMessage.st-emotion-cache-1c7y2gy:nth-child(even) { /* This targets the user message, which is typically the second (even) child in the message list */
    background-color: #e0f2f7; /* Light blue for user messages */
    color: #212121;
    align-self: flex-end; /* Align user messages to the right */
    border-bottom-right-radius: 2px;
    margin-left: auto; /* Push to the right */
    border: 1px solid #cceeff;
}

/* Assistant message bubble */
.stChatMessage.st-emotion-cache-1c7y2gy:nth-child(odd) { /* This targets the assistant message, typically the first (odd) child */
    background-color: #f7f7f7; /* Light grey for assistant messages */
    color: #333;
    align-self: flex-start; /* Align assistant messages to the left */
    border-bottom-left-radius: 2px;
    margin-right: auto; /* Push to the left */
    border: 1px solid #eaeaea;
}

/* Chat input bar styling */
.stTextInput > div > div > input {
    border-radius: 25px;
    padding: 10px 20px;
    border: 1px solid #ccc;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    font-size: 1rem;
}

/* Send button styling */
.stButton button {
    background-color: #007bff; /* Blue button */
    color: white;
    border-radius: 25px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton button:hover {
    background-color: #0056b3; /* Darker blue on hover */
}

/* Spinner styling */
.stSpinner > div > div {
    color: #007bff; /* Spinner color to match theme */
}

/* Adjust title font */
h1 {
    color: #007bff;
    text-align: center;
    font-weight: 600;
}

/* Styling for the new caption container */
.caption-container {
    background-color: #e6f7ff; /* Light blue background */
    border: 1px solid #cceeff; /* Light blue border */
    border-radius: 8px; /* Slightly rounded corners */
    padding: 0.75rem 1.25rem; /* Padding inside the container */
    margin-top: 1.5rem; /* Space above it */
    margin-bottom: 2rem; /* Space below it */
    text-align: center;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); /* Subtle shadow */
    width: fit-content; /* Make container fit its content width */
    max-width: 90%; /* Max width to keep it from being too wide */
    margin-left: auto; /* Center the block */
    margin-right: auto; /* Center the block */
}

/* Styling for the caption text itself, within the container */
.caption-container p { /* Target the paragraph inside st.markdown */
    color: #336699; /* Darker blue text for contrast */
    font-size: 1.1em; /* Increased font size for better visibility */
    margin: 0; /* Remove default paragraph margin */
    line-height: 1.4;
}

/* Clear Chat button specific styling for smaller width */
/* IMPORTANT: This class needs to be applied via JS as direct class assignment is not natively supported by Streamlit buttons */
.small-button-style {
    width: auto; /* Adjust width based on content */
    padding: 8px 15px; /* Smaller padding */
    font-size: 0.85rem; /* Smaller font */
    border-radius: 20px; /* Slightly smaller border radius */
    background-color: #dc3545 !important; /* Ensure red background */
    color: white !important; /* Ensure white text */
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

/* Ensure messages fill container and are scrollable if needed */
.st-emotion-cache-zt5ig8 { /* This targets the container holding the chat messages */
    overflow-y: auto;
    max-height: 60vh; /* Adjust as needed */
    padding-right: 15px; /* Space for scrollbar */
}

/* Remove default Streamlit footer */
footer { visibility: hidden; }

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
            message_placeholder.markdown(full_response)

        else:
            if conversation_chain:
                try:
                    response_obj = conversation_chain.invoke(
                        {"question": prompt, "chat_history": st.session_state.conversation_memory.buffer_as_messages}
                    )
                    full_response = response_obj["answer"]
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"An error occurred while getting a response: {e}. Please try again."
                    st.error(full_response)
            else:
                full_response = "Chatbot is not fully initialized. Please check the backend configuration."
                st.markdown(full_response)

    # Add assistant message to chat history for future context
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Clear Chat Button (Moved to bottom and styled for smaller size) ---
# Removed the `st.info` note as requested.
# Using a key to differentiate this button and applying a custom class via JS
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