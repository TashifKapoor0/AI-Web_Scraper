# 🔎 AI Web Scraper Chatbot

An AI-powered Streamlit application that scrapes content from any public website URL, structures the extracted content into organized sections, and allows users to download the clean, structured output — powered by **Azure OpenAI GPT-4o**, **text embeddings**, and **Azure Cosmos DB** for chat history persistence.

---

## 📌 Features

- 🌐 Scrapes visible content from public web pages.
- ✂️ Removes unnecessary elements: scripts, headers, footers, nav menus, ads, and cookie notices.
- 📑 Groups content by visible headings (e.g. OVERVIEW, EVENT DETAILS).
- ⚡ Cleaned and structured using **GPT-4o** via Azure OpenAI.
- 💾 Saves entire conversation and scraping history in **Azure Cosmos DB**.
- 📥 Download structured results as **.txt** and **.json** files.
- 📊 Streamlit-based interactive UI.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Azure OpenAI (GPT-4o + Embeddings)**
- **Azure Cosmos DB**
- **BeautifulSoup & Requests**
- **tiktoken**
- **pandas**

---

## 📦 Installation

1️⃣ Clone the repository:
git clone https://github.com/yourusername/ai-web-scraper-chatbot.git
cd ai-web-scraper-chatbot

2️⃣ Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate 

3️⃣ Install dependencies:
pip install -r requirements.txt

4️⃣ Create a .env file 

5️⃣ Run the application:
streamlit run app.py
