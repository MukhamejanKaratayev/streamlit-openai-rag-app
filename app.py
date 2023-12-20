import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Document Q&A with AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Utility functions


def get_conversation_chain(vectorestore):
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key='chat_history',
        return_message=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=temperature_input, model_name=model_select),
        memory=memory,
        retriever=vectorestore.as_retriever(),
        get_chat_history=lambda h: h,
    )
    return conversation_chain


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorestore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorestore


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    return text_splitter.split_text(raw_text)


def get_pdf_text(get_pdf_text):
    text = ""
    for file in get_pdf_text:
        try:
            pdf_reader = PdfReader(file)
        except (PdfReader.PdfReadError, pyPDF2.utils.PdfReadError) as e:
            st.error(f"Ошибка при загрузке файла {file} : {e}")
            continue

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                print(f"Страница {page} не содержит текста")
    return text

# Sidebar


with st.sidebar:
    st.subheader('Выбор модели')
    llm_model_oprions = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'gpt-4']
    model_select = st.selectbox(
        'Выберите LLM модель',
        llm_model_oprions,
        index=0
    )
    st.markdown('\n')
    temperature_input = st.slider(
        'Задайте рандомность LLM модели',
        min_value=0.0,
        max_value=1.0,
        value=0.4,
    )
    st.markdown('\n')

    st.divider()

    st.subheader('База знаний')
    user_uploads = st.file_uploader(
        'Загрузите файл с текстом',
        type=['pdf'],
        accept_multiple_files=True,
    )

    if user_uploads is not None:
        if st.button('Загрузить', use_container_width=True):
            with st.spinner('Обработка...'):
                # get text from pdf
                raw_text = get_pdf_text(user_uploads)

                # split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # convert chunks to vectors and store in memory
                vectorestore = get_vectorstore(text_chunks)

                # create langchain chain
                st.session_state.conversation = get_conversation_chain(
                    vectorestore)

    clear_history = st.button(
        "Очистить историю", use_container_width=True
    )

st.title("📑 RAG app for Document Q&A with AI")


if 'doc_messages' not in st.session_state or clear_history:
    # Start with first message from assistant
    st.session_state['doc_messages'] = [
        {"role": "assistant", "content": "Здравствуйте! Я ваш ИИ-помощник. Готов помочь вам с анализом ваших документов."}
    ]
    # Initialize chat_history as an empty list
    st.session_state['chat_history'] = []


# Display previous chat messages
for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# User query input and processing
if user_query := st.chat_input("Введите ваш вопрос:"):

    # Add user's message to chat history
    st.session_state['doc_messages'].append(
        {"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Думаю..."):
        # Check if the conversation chain is initialized
        if 'conversation' in st.session_state:
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            # Process the user's message using the conversation chain
            result = st.session_state.conversation({
                "question": user_query,
                "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            # Append the user's question and AI's answer to chat_history
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response
            })
        else:
            response = "Please upload a document first to initialize the conversation chain."

        # Display AI's response in chat format
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add AI's response to doc_messages for displaying in UI
        st.session_state['doc_messages'].append(
            {"role": "assistant", "content": response})
