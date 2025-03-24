import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import io
import nltk
import os

# Download NLTK data
nltk.download("punkt")

# Streamlit configuration
st.set_page_config(page_title="NewsGPT: AI-Powered News Analysis", page_icon="📰")

# Set up the API key for LangChain Google Generative AI
os.environ["GOOGLE_API_KEY"] = st.secrets.get("api_key", "")

# Title and navigation bar
st.title("Stay Informed with AI-Powered News Analysis")
st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Go to", ["Home", "News Categories", "Ask Questions"])

# Initialize session state
if "URLS_INPUT" not in st.session_state:
    st.session_state.URLS_INPUT = []
if "check" not in st.session_state:
    st.session_state.check = False
if "vectorindex_openai" not in st.session_state:
    st.session_state.vectorindex_openai = None

# Helper functions for live news
def fetch_news_search_topic(topic):
    site = f"https://news.google.com/rss/search?q={topic}"
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, "xml")
    return sp_page.find_all("item")


def fetch_top_news():
    site = "https://news.google.com/news/rss"
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, "xml")
    return sp_page.find_all("item")


def display_news(news_list, news_quantity):
    for i, news in enumerate(news_list[:news_quantity], start=1):
        news_data = Article(news.link.text)
        news_data.download()
        news_data.parse()
        news_data.nlp()
        st.markdown(f"### {news.title.text}")
        st.image(news_data.top_image if news_data.top_image else "https://via.placeholder.com/300")
        st.write(news_data.summary)
        st.markdown(f"[Read more...]({news.link.text})")
        st.markdown("---")


# Helper functions for Q&A
def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorindex = FAISS.from_documents(docs, embeddings)
    vectorindex.save_local("faiss_index")

    return vectorindex


def query_model(question):
    vectorindex_openai = FAISS.load_local(
        "faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())
    response = chain({"question": question}, return_only_outputs=True)
    return response


# Navigation logic
if navigation == "Home":
    st.header("Trending News")
    st.subheader("🔥 Latest News")
    news_count = st.slider("Number of news articles:", 1, 10, 5)
    top_news = fetch_top_news()
    display_news(top_news, news_count)

elif navigation == "News Categories":
    st.header("News by Categories")
    categories = ["World", "Business", "Technology", "Sports"]
    selected_category = st.selectbox("Select a category", categories)
    if selected_category:
        st.subheader(f"📰 {selected_category} News")
        news_count = st.slider("Number of news articles:", 1, 10, 5)
        category_news = fetch_news_search_topic(selected_category)
        display_news(category_news, news_count)

elif navigation == "Ask Questions":
    st.header("Ask Questions about Uploaded URLs")
    st.sidebar.subheader("Add URLs for Processing")
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_input_{i}")
        if url:
            st.session_state.URLS_INPUT.append(url)

    if st.sidebar.button("Process URLs"):
        st.session_state.vectorindex_openai = process_urls(st.session_state.URLS_INPUT)
        st.session_state.check = True
        st.success("Processing complete. You can now ask questions.")

    if st.session_state.check:
        question = st.text_input("Enter your question:")
        if question:
            response = query_model(question)
            st.markdown(f"**Answer:** {response['answer']}")
            st.markdown(f"**Sources:** {response['sources']}")
