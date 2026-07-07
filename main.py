import pip_system_certs.wrapt_requests
import os
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title(" News Research Tool ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_path = "faiss_index_gemini"
main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.9,
    max_output_tokens=500,
    timeout=30
)

if process_url_clicked:
    # Filter out empty URL fields
    valid_urls = [u for u in urls if u.strip() != ""]

    if len(valid_urls) == 0:
        st.error("Please enter at least one URL before processing.")
        st.stop()

    loader = UnstructuredURLLoader(urls=valid_urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    if len(data) == 0:
        st.error(
            "No content could be loaded from the URL(s) provided. "
            "The site may be blocking scrapers or require JavaScript. "
            "Try different URLs."
        )
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    if len(docs) == 0:
        st.error(
            "The URLs loaded, but no usable text content was found after splitting. "
            "Try different URLs."
        )
        st.stop()

    # Build a NEW FAISS index from the freshly scraped documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save using FAISS's own method (avoids the pickle thread-lock error)
    vectorstore_gemini.save_local(index_path)

    main_placeholder.text("Processing complete! You can now ask a question below. ✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_path):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        st.write("Building chain...")
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        st.write("Chain built. Sending request to Gemini now...")

        try:
            result = chain.invoke({"question": query}, return_only_outputs=True)
            st.write("Got a response back!")
        except Exception as e:
            st.error(f"Error during chain execution: {e}")
            st.stop()

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.error("No processed data found. Please process some URLs first using the sidebar.")