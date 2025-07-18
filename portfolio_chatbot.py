import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (for API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file or Streamlit secrets.")
    st.stop()

# --- Initialize Google Generative AI components in global scope ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    max_output_tokens=400,
    streaming=True
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


# --- UPDATED: Significantly Expanded Knowledge Base with Education Details ---
dummy_texts = [
    # --- About Me / Overview ---
    "Nithin Shetty M is an ambitious and results-driven AI/ML Engineering student pursuing a Bachelor of Engineering at Vivekananda College of Engineering & Technology (VCET) Puttur, specializing in Artificial Intelligence and Machine Learning.",
    "He is deeply passionate about leveraging cutting-edge AI and Machine Learning technologies to develop innovative solutions for complex real-world problems.",
    "Nithin combines strong theoretical knowledge with practical project experience in areas like deep learning, natural language processing, computer vision, and data analysis.",
    "He is actively seeking challenging full-time opportunities or internships in AI/ML Engineering, Data Science, or Software Development roles.",
    "Nithin is a quick learner, highly adaptable, and thrives in collaborative environments, always eager to contribute to impactful projects.",

    # --- Education ---
    "Nithin Shetty M is currently a B.E. student in Artificial Intelligence and Machine Learning at Vivekananda College of Engineering & Technology (VCET), Puttur, with an expected graduation in 2026.",
    "His academic curriculum at VCET includes advanced topics such as Machine Learning Algorithms, Deep Learning Architectures, Natural Language Processing, Computer Vision, Data Structures & Algorithms, and Software Engineering Principles.",
    "Nithin maintains a strong academic record, demonstrating his commitment to mastering core AI/ML concepts.",
    "He has actively participated in various workshops and seminars related to emerging AI trends and technologies.",
    # New: PUC (Pre-University Course) details
    "Nithin completed his Pre-University Course (PUC) at [Your PUC College Name], from [Year Start] to [Year End], specializing in [Your PUC Stream, e.g., PCMB - Physics, Chemistry, Mathematics, Biology or PCMC - Physics, Chemistry, Mathematics, Computer Science].",
    "During his PUC, Nithin achieved [Your PUC Percentage/Grade, e.g., 88% distinction].",
    # New: SSLC (Secondary School Leaving Certificate) details
    "For his secondary education, Nithin completed his SSLC at [Your SSLC School Name] in [Year of Completion], where he demonstrated strong foundational knowledge across subjects.",
    "Nithin scored [Your SSLC Percentage/Grade, e.g., 96.36% with distinction] in his SSLC examinations.",

    # --- Skills ---
    # Programming Languages
    "Nithin is highly proficient in Python, a primary language for his AI/ML and data science projects.",
    "He has solid programming skills in Java, used for backend development and algorithmic problem-solving.",
    # Machine Learning & Deep Learning Frameworks
    "His expertise includes TensorFlow and Keras for building and deploying deep learning models.",
    "Nithin is also proficient with PyTorch, used for advanced research and development in deep learning.",
    "He uses Scikit-learn extensively for classical machine learning algorithms, model training, and evaluation.",
    # Data Science & Analysis Tools
    "For data manipulation and analysis, Nithin leverages Pandas and NumPy, essential for preprocessing and numerical operations.",
    "He is skilled in data visualization using libraries like Matplotlib and Seaborn to uncover insights from complex datasets.",
    # Other Technical Skills
    "Nithin has strong foundations in Natural Language Processing (NLP) for text analysis, sentiment analysis, and conversational AI.",
    "He is experienced in Computer Vision, including image processing, object detection, and facial recognition techniques.",
    "His database skills include working with SQL (MySQL, PostgreSQL) for data storage and retrieval.",
    "Nithin is proficient with Git and GitHub for version control, collaborative development, and managing project repositories.",
    "He has experience building interactive web applications using Streamlit for data dashboards and AI demos.",
    "Familiarity with cloud platforms like AWS or Google Cloud Platform (GCP) for deploying machine learning models and services.",
    "Nithin has problem-solving abilities showcased through competitive programming challenges and project development.",

    # --- Projects (Detailed descriptions with Technologies and Contributions) ---

    # Project 1: AIRA Teaching Bot
    "Project Title: AIRA Teaching Bot",
    "Description: The AIRA Teaching Bot is an advanced AI-powered educational assistant designed to revolutionize interactive learning experiences for students.",
    "Nithin's Role: Nithin was the lead developer responsible for the core conversational AI logic, leveraging Natural Language Processing (NLP) techniques and Large Language Models (LLMs).",
    "Technologies: Python, Hugging Face Transformers, TensorFlow, Streamlit (for UI), Flask (for API), NLTK.",
    "Features: Provides personalized explanations, generates quizzes dynamically, offers real-time feedback, and tracks student progress. It aims to make learning more engaging and accessible.",
    "Impact: Pilot studies demonstrated a 30% improvement in student engagement and understanding.",
    "GitHub Link for AIRA Teaching Bot: https://github.com/nithinshettygit/AIRA-Teaching-Bot (Replace with actual link if available)",

    # Project 2: Autonomous Wheelchair
    "Project Title: Autonomous Wheelchair",
    "Description: An innovative project focusing on developing a smart wheelchair capable of autonomous navigation and real-time obstacle avoidance.",
    "Nithin's Role: Nithin implemented deep learning models for environment perception and decision-making, integrating sensor data for robust navigation.",
    "Technologies: Python, TensorFlow, OpenCV, Raspberry Pi, various sensors (ultrasonic, LiDAR).",
    "Features: Equipped with intelligent pathfinding algorithms, capable of navigating complex indoor and outdoor environments, and ensuring user safety through obstacle detection.",
    "Impact: Aims to provide greater independence for individuals with mobility challenges, enhancing their quality of life.",
    "GitHub Link for Autonomous Wheelchair: https://github.com/nithinshettygit/Autonomous-Wheelchair (Replace with actual link if available)",

    # Project 3: Hand Gesture Controlled Wheelchair
    "Project Title: Hand Gesture Controlled Wheelchair",
    "Description: This project demonstrates intuitive control of a wheelchair using real-time hand gesture recognition through computer vision.",
    "Nithin's Role: Nithin designed and implemented the computer vision pipeline for accurate gesture detection and mapping gestures to wheelchair movements.",
    "Technologies: Python, OpenCV, MediaPipe, Arduino (for motor control).",
    "Features: Allows users to control wheelchair direction (forward, backward, left, right, stop) with simple hand movements, offering an alternative control interface.",
    "Impact: Provides an accessible and user-friendly control mechanism, particularly beneficial for users who may have difficulty with traditional joysticks.",
    "GitHub Link for Hand Gesture Controlled Wheelchair: https://github.com/nithinshettygit/Hand-Gesture-Controlled-Wheelchair (Replace with actual link if available)",

    # --- Experience (If applicable, add specific roles, dates, responsibilities) ---
    "Nithin's professional experience includes internships where he applied his AI/ML skills in practical settings.",
    "He has contributed to various team projects, showcasing his collaboration and problem-solving abilities.",

    # --- Contact and Social Media ---
    "You can connect with Nithin Shetty M through several professional platforms.",
    "His primary professional networking platform is LinkedIn. You can find his full profile here: https://www.linkedin.com/in/nithin-shetty-m-8646b3226/",
    "Explore Nithin's code repositories and project implementations on his GitHub profile: https://github.com/nithinshettygit",
    "For direct communication regarding job opportunities or collaborations, Nithin's email is: shettyn517@gmail.com",
    "Nithin is always open to discussing new ideas and opportunities in the AI/ML space. Feel free to reach out!"
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


# --- Define the custom prompt for the chatbot ---
CUSTOM_PROMPT_TEMPLATE = """You are a helpful and concise AI assistant for Nithin Shetty M's portfolio.
Answer the user's questions truthfully and specifically based ONLY on the provided context about Nithin.
If a question is about Nithin's skills, projects, education, or contact details, provide a direct answer.
If a link (GitHub, LinkedIn, email) is mentioned in the context, always include the full link in your response if relevant to the question.
Do NOT make up answers or provide information not present in the context.
If the information is not directly in the context, politely state that you cannot answer from the available information, but suggest common topics like 'skills', 'projects', 'education', or 'contact'.

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
def get_conversation_chain(_llm_model: ChatGoogleGenerativeAI, _vector_store: FAISS, _memory: ConversationBufferWindowMemory):
    if _vector_store is None:
        return None
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_model,
            retriever=_vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant documents
            memory=_memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=False
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

.stCaption {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}

/* Clear Chat button */
.stButton:last-of-type button { /* Target the clear chat button specifically */
    background-color: #dc3545; /* Red for clear button */
    margin-top: 1rem;
    width: 100%;
}

.stButton:last-of-type button:hover {
    background-color: #c82333;
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


st.title("💬 Nithin's AI Portfolio Assistant")
st.caption("Ask me anything about Nithin Shetty M's projects, skills, and experience!")

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
                    # LangChain's .stream() method yields chunks
                    for chunk in conversation_chain.stream({"question": prompt, "chat_history": st.session_state.conversation_memory.buffer_as_messages}):
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                        elif isinstance(chunk, AIMessage):
                            full_response += chunk.content
                        elif isinstance(chunk, dict) and "content" in chunk:
                            full_response += chunk["content"]
                        else:
                            full_response += str(chunk)

                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"An error occurred while getting a response: {e}. Please try again."
                    st.error(full_response)
            else:
                full_response = "Chatbot is not fully initialized. Please check the backend configuration."
                st.markdown(full_response)

    # Add assistant message to chat history for future context
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Disclaimer for dummy data
st.markdown("---")
st.info("Note: This chatbot provides information based on Nithin's portfolio data. For comprehensive details, please refer to his full portfolio sections or direct contact information.")

# Button to clear the chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        k=5
    )
    st.rerun()
