# 🗞️ Instant Insight: AI-Powered News Summarization Engine  

Welcome to **Instant Insight**, your intelligent news companion that summarizes the latest stories in real-time using advanced AI. Powered by LLMs and NLP, this project helps you stay informed without getting overwhelmed.

![Status](https://img.shields.io/badge/status-in--progress-yellow)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

The web is full of breaking news, hot takes, and endless articles. But who has time to read them all?  
**Instant Insight** solves this with AI-driven summarization that captures the essence of the news in seconds — so you can focus on what matters.

It combines:
- 🧠 **LLMs for abstractive summarization**
- ✂️ **Rule-based extractive summarization**
- 📊 **Real-time data scraping**
- 🎛️ **An interactive, responsive UI**

---

## 🎯 Key Features

- ✅ **Real-Time News Scraping** – Fetches articles from Google News RSS feeds.
- ✅ **Hybrid Summarization** – Uses both extractive (Newspaper3k) and abstractive (GPT) summarization methods.
- ✅ **LLM Integration** – Incorporates GPT-based models to generate human-like summaries.
- ✅ **Interactive Streamlit UI** – Expandable cards, category filters, and search by topic.
- 🔜 **Sentiment Analysis** – Detect user tone and tag articles (positive/neutral/negative).
- 🛣️ **Misinformation Detection** – Planned for future development.

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Scraping**: `BeautifulSoup`, `Newspaper3k`  
- **Extractive NLP**: `NLTK`  
- **Abstractive NLP**: `transformers`, `OpenAI`  

---

## 🚀 Try It Out

Clone the project and run locally:

```bash
git clone https://github.com/pranati10/Instant-Insight.git
cd Instant-Insight
pip install -r requirements.txt
streamlit run app.py
