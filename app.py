import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import uuid
import pandas as pd
import json
import os
import tiktoken
from io import BytesIO

# ===== Streamlit Config =====
st.set_page_config(page_title="AI Web Scraper Chatbot", layout="wide")

# ===== Azure Config =====
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")
COSMOS_PARTITION_KEY = os.getenv("COSMOS_PARTITION_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
OPENAI_API_VERSION = "2024-03-01-preview"

# ===== Initialize Azure Clients =====
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.create_database_if_not_exists(id=COSMOS_DB_NAME)
container = database.create_container_if_not_exists(
    id=COSMOS_CONTAINER_NAME,
    partition_key=PartitionKey(path=COSMOS_PARTITION_KEY)
)

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=OPENAI_API_VERSION
)

# ===== Chat Setup =====
session_id = str(uuid.uuid4())
chat_history = []

# ===== GPT System Prompt =====
system_message = """
You are a data extraction assistant.

Your task is to extract and structure content from raw HTML copied from a public website.

Rules:
- DO NOT summarize, rewrite, or infer anything.
- Preserve ALL meaningful visible content.
- Remove all non-content elements: scripts, nav menus, cookie banners, footers, ads, accessibility or legal notices, etc.
- Group content by visible headings (e.g., OVERVIEW, EVENT DETAILS, FAQ, etc.)
- Keep line breaks, bullet points, and original wording.
- Use this format:

=== HEADING TITLE ===
(Original content...)

- Return plain text only. No HTML, JSON, or explanations.
"""

# ===== Web Scraper Function (Improved for Content) =====
def extract_visible_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "header", "footer", "nav", "form", "noscript", "svg", "aside", "dialog", "iframe"]):
            tag.decompose()

        # Extract social media links
        social_domains = ["facebook.com", "twitter.com", "linkedin.com", "instagram.com", "youtube.com", "t.me", "wa.me"]
        social_links = set()
        for a in soup.find_all("a", href=True):
            if any(domain in a["href"] for domain in social_domains):
                social_links.add(a["href"].strip())

        content_blocks = []
        current_heading = None
        block = []

        for element in soup.body.descendants:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_heading and block:
                    content_blocks.append(f"=== {current_heading.get_text(strip=True).upper()} ===\n{' '.join(block).strip()}")
                current_heading = element
                block = []
            elif element.name in ['p', 'li', 'div', 'span']:
                text = element.get_text(strip=True)
                if text:
                    block.append(text)

        if current_heading and block:
            content_blocks.append(f"=== {current_heading.get_text(strip=True).upper()} ===\n{' '.join(block).strip()}")
        elif block:
            content_blocks.append('\n'.join(block).strip())

        # Add social links at the end
        if social_links:
            links_formatted = "\n".join(sorted(social_links))
            content_blocks.append(f"=== SOCIAL MEDIA LINKS ===\n{links_formatted}")

        visible_text = "\n\n".join(filter(None, content_blocks))
        return visible_text

    except requests.exceptions.RequestException as e:
        return f"ERROR: Failed to scrape the page: {e}"
    except Exception as e:
        return f"ERROR: An unexpected error occurred during scraping: {e}"

# ===== GPT Cleaning Function =====
def clean_and_structure_text(raw_text):
    model_name = GPT_DEPLOYMENT_NAME
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding for many OpenAI models

    max_tokens_input = 120000  # Leave some buffer for the system prompt and response
    encoded_text = encoding.encode(raw_text)

    if len(encoded_text) > max_tokens_input:
        truncated_tokens = encoded_text[:max_tokens_input]
        truncated_text = encoding.decode(truncated_tokens)
        st.warning(f"The scraped content was too long and has been truncated ({len(encoded_text) - len(truncated_tokens)} tokens removed) before sending to GPT.")
    else:
        truncated_text = raw_text

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": truncated_text}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: GPT processing failed: {e}"

# ===== Save Chat to Cosmos =====
def save_to_cosmos():
    container.upsert_item({
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "chat": chat_history
    })

# ===== Download Options =====
def create_download_buttons(content):
    st.download_button("Download as .txt", content, file_name="structured.txt")

    json_data = {"structured_content": content}
    st.download_button("Download as .json", json.dumps(json_data, indent=2), file_name="structured.json")

# ===== Streamlit UI (Improved) =====
st.title("🔎 AI Web Scraper Chatbot")
st.markdown("""
Paste any **public website URL** below.  
I'll scrape the content and return a **clean, structured format** grouped by section headings.
""")

# Persistent input and result
if "last_url" not in st.session_state:
    st.session_state.last_url = ""
if "response" not in st.session_state:
    st.session_state.response = ""

# Input Area
with st.container():
    user_input = st.text_input("🌐 Enter Website URL:", value=st.session_state.last_url)

    if st.button("🚀 Extract Content"):
        st.session_state.last_url = user_input
        chat_history.append({"role": "user", "content": user_input})

        if not user_input.startswith("http"):
            response = "❌ Please enter a valid URL starting with http or https."
            st.error(response)
        else:
            with st.spinner("🔄 Scraping website..."):
                raw = extract_visible_text(user_input)

            if raw.startswith("ERROR"):
                response = raw
                st.error(response)
            else:
                with st.spinner("🤖 Structuring content..."):
                    response = clean_and_structure_text(raw)
                st.success("✅ Content structured successfully!")
                create_download_buttons(response)

        chat_history.append({"role": "bot", "content": response})
        st.session_state.response = response
        save_to_cosmos()

# Output Section
if st.session_state.response:
    with st.expander("📝 Structured Content (Click to Expand)", expanded=True):
        st.code(st.session_state.response, language="markdown")
      
