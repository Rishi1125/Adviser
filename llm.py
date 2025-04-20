# --- START OF FILE ft_llm.py ---

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import re
# import json # No longer used directly
import os
from datetime import datetime # , timedelta # No longer used directly
import urllib.parse
# from PIL import Image # No longer used directly
# from io import BytesIO # No longer used directly
import time
import nest_asyncio
from tqdm import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import asyncio
import traceback # For detailed error logging

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Assitant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = None
if 'last_analyzed' not in st.session_state:
    st.session_state.last_analyzed = None
if 'gpt2_model' not in st.session_state:
    st.session_state.gpt2_model = None
if 'gpt2_tokenizer' not in st.session_state:
    st.session_state.gpt2_tokenizer = None
if 'stock_predictions' not in st.session_state:
    st.session_state.stock_predictions = None

# --- Utility Functions ---

def clean_text(text):
    """Clean text from HTML and normalize whitespace"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_large_number(num, currency=True, exchange="US"):
    """Format large numbers with appropriate suffixes and currency symbols"""
    if num == 'N/A' or num is None or not isinstance(num, (int, float)):
        return 'N/A'

    suffix = ''
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        num /= 1_000_000_000_000
        suffix = 'T'
    elif abs_num >= 1_000_000_000:
        num /= 1_000_000_000
        suffix = 'B'
    elif abs_num >= 1_000_000:
        num /= 1_000_000
        suffix = 'M'
    elif abs_num >= 1_000:
        num /= 1_000
        suffix = 'K'

    num_str = f"{num:.2f}{suffix}"

    if currency:
        if exchange in ["NSE", "BSE"]:
            return f"₹{num_str}"
        else:
            # Default to USD, can be expanded later if needed
            return f"${num_str}"
    else:
        return num_str

def download_with_retry(url, dest_path, max_retries=3, timeout=30):
    """Download a file with retry logic for timeouts."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            with open(dest_path, 'wb') as file, tqdm(
                    desc=f"Downloading (attempt {attempt+1}/{max_retries})",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)

            return True
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                st.warning(f"Download attempt {attempt+1} failed: {str(e)}. Retrying...")
                time.sleep(2) # Wait before retrying
            else:
                st.error(f"Download failed after {max_retries} attempts: {str(e)}")
                return False
        except requests.exceptions.RequestException as e:
             st.error(f"Download failed: {str(e)}")
             return False

def ensure_model_is_cached(model_name):
    """Check if model is cached and download if necessary using huggingface_hub."""
    try:
        # Check if model is already cached locally
        # Construct a potential path (this might vary slightly based on HF cache structure)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_cache_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")

        # If the directory exists and isn't empty, assume it's cached
        if os.path.exists(model_cache_path) and os.listdir(model_cache_path):
             # st.info(f"Model '{model_name}' found in cache.") # Reduce verbosity
             return True

        st.info(f"Model '{model_name}' not found in cache. Attempting download...")
        # Try to download/ensure the model is cached
        snapshot_download(
            repo_id=model_name,
            # local_dir=model_cache_path, # snapshot_download manages the cache dir
            local_dir_use_symlinks=False, # Recommended for Windows compatibility
            resume_download=True, # Try to resume if interrupted
            # max_workers=4 # Use multiple workers if desired
        )
        st.success(f"Model '{model_name}' downloaded successfully.")
        return True
    except HfHubHTTPError as e:
        st.error(f"Error downloading model '{model_name}': {str(e)}")
        st.error("Please check your internet connection and Hugging Face Hub status.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while checking/downloading the model '{model_name}': {str(e)}")
        return False

# --- Financial NLP with FinBERT ---

class FinancialSentimentAnalyzer(nn.Module):
    def __init__(self, n_classes, pretrained_model='ProsusAI/finbert'):
        super(FinancialSentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        output = self.fc(output)
        return self.softmax(output)

class FinancialNLP:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        # st.info(f"Using device: {self.device}") # Reduce verbosity

        # Wrapped in function scope to avoid re-running on every interaction if cached
        if 'finbert_loaded' not in st.session_state:
             st.session_state.finbert_loaded = False

        if not st.session_state.finbert_loaded:
             with st.spinner("Loading model..."):
                try:
                    # Load tokenizer
                    self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
                    # Initialize the model architecture
                    self.model = FinancialSentimentAnalyzer(n_classes=3).to(self.device)

                    if model_path and os.path.exists(model_path):
                         try:
                             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                             st.success(f"Custom weights loaded from {model_path}")
                         except Exception as load_err:
                             st.error(f"Error loading custom weights from {model_path}: {str(load_err)}")
                             st.warning("Loading...")
                             self.model = FinancialSentimentAnalyzer(n_classes=3).to(self.device)
                    else:
                         if model_path: st.warning(f"Path '{model_path}' not found. Using base FinBERT.")
                         else: st.info("Loading...")

                    self.model.eval()
                    st.session_state.finbert_loaded = True # Mark as loaded
                    # st.success("FinBERT loaded.") # Reduce verbosity

                except OSError as e:
                     st.error(f"Error loading FinBERT model/tokenizer: {str(e)}")
                     self.model, self.tokenizer = None, None
                except Exception as e:
                    st.error(f"Unexpected error during FinBERT init: {str(e)}")
                    self.model, self.tokenizer = None, None
        else:
             # Already loaded in session, just re-assign if necessary
             # This part might need refinement depending on exact caching behavior
             pass

    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            # st.warning("Sentiment model not loaded. Returning neutral.")
            return 1 # Neutral default

        self.model.eval()
        cleaned_text = clean_text(text)
        if not cleaned_text: return 1

        try:
            encoding = self.tokenizer.encode_plus(
                cleaned_text,
                add_special_tokens=True, max_length=128,
                return_token_type_ids=False, padding='max_length',
                truncation=True, return_attention_mask=True, return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
            return predicted.item()
        except Exception as e:
            # st.warning(f"Error during sentiment prediction for text: '{text[:50]}...'. Error: {e}") # Avoid excessive logging
            return 1 # Default to neutral on error


    def analyze_sentiment(self, headlines):
        if not headlines or self.model is None or self.tokenizer is None:
            return None

        # Batch prediction (can be faster for many headlines, but complicates error handling per headline)
        # For simplicity, sticking to individual predictions first.

        sentiments = [self.predict(headline) for headline in headlines]

        if not sentiments: return None # All predictions might have failed

        try: # Add try-except for calculations
             avg_sentiment_score = sum(sentiments) / len(sentiments)
             sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
             sentiment_counts = {
                 "Positive": sentiments.count(2),
                 "Neutral": sentiments.count(1),
                 "Negative": sentiments.count(0)
             }
             if avg_sentiment_score > 1.2: overall_sentiment = "positive"
             elif avg_sentiment_score < 0.8: overall_sentiment = "negative"
             else: overall_sentiment = "neutral"
             sentiment_out_of_10 = min(10, max(1, round(1 + (avg_sentiment_score / 2) * 9)))

             return {
                 "overall_sentiment": overall_sentiment,
                 "sentiment_score": sentiment_out_of_10,
                 "headline_sentiments": list(zip(headlines, [sentiment_map[s] for s in sentiments])),
                 "sentiment_counts": sentiment_counts,
                 "avg_sentiment_raw": avg_sentiment_score
             }
        except Exception as e:
            st.warning(f"Error calculating overall sentiment: {e}")
            return None # Return None if calculation fails


# --- GPT-2 Model Loading ---

@st.cache_resource
def load_gpt2_model():
    # st.info("Loading GPT-2 model and tokenizer...") # Reduce verbosity
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # st.info(f"Using device for GPT-2: {device}")

        # Ensure model is cached
        if not ensure_model_is_cached("gpt2"):
            st.error("Failed to download or cache GPT-2 model.")
            return None, None, None

        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # st.info("Set GPT-2 pad_token to eos_token.")

        # Load model
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.eval()
        # st.success("GPT-2 model and tokenizer loaded.") # Reduce verbosity
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {str(e)}")
        return None, None, None

# --- Stock Predictions Data Handling ---

@st.cache_data
def load_stock_predictions(file_path='next_30_days_predictions.csv'):
    # st.info(f"Loading stock predictions from '{file_path}'...") # Reduce verbosity
    try:
        df = pd.read_csv(file_path)
        # Basic validation
        if 'Stock' not in df.columns or 'Date' not in df.columns or 'Predicted Price' not in df.columns:
             st.error(f"Prediction file '{file_path}' missing required columns (Stock, Date, Predicted Price).")
             return pd.DataFrame() # Return empty DataFrame
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as date_err:
            st.warning(f"Could not parse 'Date' column in '{file_path}'. Error: {date_err}")
            return pd.DataFrame()
        if 'Accuracy' not in df.columns:
             # st.warning(f"'Accuracy' column not found in '{file_path}'. Treating as N/A.")
             df['Accuracy'] = np.nan
        # st.success(f"Loaded {len(df)} predictions for {df['Stock'].nunique()} stocks.") # Reduce verbosity
        return df
    except FileNotFoundError:
        st.warning(f"Stock prediction file not found: '{file_path}'. Prediction features disabled.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading stock predictions from '{file_path}': {str(e)}")
        return pd.DataFrame()

def get_unique_stocks():
    """Return the list of unique stocks in the prediction dataset"""
    if st.session_state.stock_predictions is None or st.session_state.stock_predictions.empty:
        return []
    return sorted(st.session_state.stock_predictions['Stock'].unique().tolist())

def get_stock_predictions(stock_name, days=7):
    """Get predictions for a specific stock for the next N days"""
    if st.session_state.stock_predictions is None or st.session_state.stock_predictions.empty:
        return None
    stock_data = st.session_state.stock_predictions[st.session_state.stock_predictions['Stock'].str.lower() == stock_name.lower()]
    if stock_data.empty: return None
    return stock_data.sort_values('Date').head(days)

def get_top_stocks_by_growth(days=7, top_n=5):
    """Get top N stocks with highest predicted growth in the next N days"""
    if st.session_state.stock_predictions is None or st.session_state.stock_predictions.empty:
        return None
    unique_stocks = get_unique_stocks()
    growth_data = []
    for stock in unique_stocks:
        stock_data = get_stock_predictions(stock, days)
        if stock_data is not None and len(stock_data) >= 2:
            first_day_price = stock_data.iloc[0]['Predicted Price']
            last_day_price = stock_data.iloc[-1]['Predicted Price']
            if first_day_price is not None and isinstance(first_day_price, (int, float)) and \
               last_day_price is not None and isinstance(last_day_price, (int, float)) and \
               abs(first_day_price) > 1e-6:
                growth_pct = ((last_day_price - first_day_price) / first_day_price) * 100
            else:
                growth_pct = 0.0
            accuracy = stock_data['Accuracy'].mean()
            accuracy = accuracy if not pd.isna(accuracy) else 'N/A'
            growth_data.append({
                'Stock': stock, 'Growth_Percentage': growth_pct, 'Start_Price': first_day_price,
                'End_Price': last_day_price, 'Accuracy': accuracy, 'Days': len(stock_data)
            })
    growth_data = sorted(growth_data, key=lambda x: x['Growth_Percentage'] if isinstance(x['Growth_Percentage'], (int, float)) else -float('inf'), reverse=True)
    return growth_data[:top_n]

# --- Chatbot Logic ---

def generate_chatbot_response(query):
    """Generate a response using the GPT-2 model and stock prediction data"""
    if st.session_state.gpt2_model is None or st.session_state.gpt2_tokenizer is None:
        return "Chatbot is initializing or failed to load. Please try again."

    dataset_answer = get_dataset_answer(query)
    if dataset_answer: return dataset_answer

    # st.info("Query not directly answerable from dataset. Using GPT-2...") # Reduce verbosity
    device = st.session_state.gpt2_model.device
    available_stocks = get_unique_stocks()
    context = f"""
    Context: Access to a dataset with 30-day stock price predictions for: {', '.join(available_stocks) if available_stocks else 'None'}.
    Can provide predictions, list stocks, show top performers, compare within dataset,answer in proper formate,give suggestion about how many quanties of a stock can be bought with their invesment plan amount and give finacial suggestion about buy stocks by analysing the provided dataset.
    User query: "{query}"
    Respond concisely based ONLY on the context and query. State if info is outside the dataset (real-time, news, unavailable stocks). .
    """
    prompt = f"Financial Assistant Prompt:\n{context}\n\nResponse:"

    try:
        inputs = st.session_state.gpt2_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = st.session_state.gpt2_model.generate(
                inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=150,
                temperature=0.7, top_p=0.9, repetition_penalty=1.2, do_sample=True,
                pad_token_id=st.session_state.gpt2_tokenizer.eos_token_id,
                eos_token_id=st.session_state.gpt2_tokenizer.eos_token_id
            )
        response = st.session_state.gpt2_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        response = re.sub(r'\.[^.]*$', '.', response) if '.' in response else response # Clean end
        if not response:
             return "Couldn't generate a specific response. Please try rephrasing."
        return response
    except Exception as e:
        st.error(f"Error during GPT-2 generation: {str(e)}")
        return "Error generating response."

def get_dataset_answer(query):
    """Try to answer the query directly using the loaded stock prediction dataset"""
    if st.session_state.stock_predictions is None or st.session_state.stock_predictions.empty:
        return "Stock prediction dataset is not loaded or is empty." # Adjusted message

    query_lower = query.lower()
    available_stocks = get_unique_stocks()

    # Intent 1: List stocks
    if any(keyword in query_lower for keyword in ["available stock", "which stock", "list of stock", "all stock", "what stock"]):
        return f"Available stocks in predictions: {', '.join(available_stocks)}" if available_stocks else "No prediction data available."

    # Intent 2: Top performers
    if ("top" in query_lower or "best" in query_lower) and any(keyword in query_lower for keyword in ["stock", "performer", "growth", "performing"]):
        days = 7; top_n = 5
        match_days = re.search(r"next (\d+) days", query_lower); match_top = re.search(r"top (\d+)", query_lower)
        if match_days: days = int(match_days.group(1))
        if match_top: top_n = int(match_top.group(1))
        top_stocks = get_top_stocks_by_growth(days=days, top_n=top_n)
        if top_stocks:
            response = f"Top {len(top_stocks)} predicted performers (next {days} days):\n"
            for i, stock in enumerate(top_stocks, 1):
                acc_str = f"{stock['Accuracy']:.2f}%" if isinstance(stock['Accuracy'], (int, float)) else "N/A"
                response += f"{i}. **{stock['Stock']}**: {stock['Growth_Percentage']:.2f}% growth (Acc: {acc_str})\n"
            return response
        else: return f"Could not determine top performers for next {days} days."

    
    # Intent 3: Compare stocks
    if any(keyword in query_lower for keyword in ["compare", "comparison", "versus", "vs"]):
        stocks_to_compare = [s for s in available_stocks if re.search(r'\b' + re.escape(s.lower()) + r'\b', query_lower)]
        if len(stocks_to_compare) >= 2:
            days = 7; match_days = re.search(r"next (\d+) days", query_lower)
            if match_days: days = int(match_days.group(1))
            
            # Use markdown table format instead of monospace formatting
            comp = f"**Stock Comparison (Next {days} Days Predictions)**\n\n"
            comp += "| Stock | Growth% | Acc% | Start₹ | End₹ |\n"
            comp += "|-------|---------|------|--------|-------|\n"
            
            found = False
            for name in stocks_to_compare:
                data = get_stock_predictions(name, days)
                if data is not None and len(data) >= 2:
                    first_p, last_p = data.iloc[0]['Predicted Price'], data.iloc[-1]['Predicted Price']
                    growth = ((last_p - first_p) / first_p) * 100 if first_p and abs(first_p)>1e-6 else 0.0
                    acc = data['Accuracy'].mean(); acc_s = f"{acc:.1f}" if not pd.isna(acc) else "N/A"
                    comp += f"| {name} | {growth:.2f} | {acc_s} | {first_p:.2f} | {last_p:.2f} |\n"
                    found = True
                else: 
                    comp += f"| {name} | N/A | N/A | N/A | N/A |\n"
            
            return comp if found else f"Could not get enough data for comparison between {', '.join(stocks_to_compare)}."


    # Intent 4: Specific stock
    found_stock = next((s for s in available_stocks if re.search(r'\b' + re.escape(s.lower()) + r'\b', query_lower)), None)
    if found_stock:
        days = 7; match_days = re.search(r"next (\d+) days", query_lower)
        if match_days: days = min(int(match_days.group(1)), 30)
        elif "month" in query_lower or "30 days" in query_lower: days=30
        data = get_stock_predictions(found_stock, days)
        if data is not None and not data.empty:
            resp = f"**Prediction: {found_stock} (Next {len(data)} Days)**\n"
            acc = data['Accuracy'].mean(); acc_s = f"{acc:.2f}%" if not pd.isna(acc) else "N/A"
            resp += f"Avg. Accuracy: {acc_s}\n"
            if len(data) >= 1:
                first_p = data.iloc[0]['Predicted Price']; first_d = data.iloc[0]['Date'].strftime('%b %d')
                last_p = data.iloc[-1]['Predicted Price']; last_d = data.iloc[-1]['Date'].strftime('%b %d')
                growth = ((last_p - first_p) / first_p) * 100 if len(data) >= 2 and first_p and abs(first_p) > 1e-6 else 0.0
                resp += f"Predicted Price Range: ₹{first_p:.2f} ({first_d}) to ₹{last_p:.2f} ({last_d})\n"
                if len(data) >= 2: resp += f"Predicted Growth: {growth:.2f}%\n"
            resp += "\n**Daily Prices:**\n" + "".join([f"- {r['Date'].strftime('%b %d')}: ₹{r['Predicted Price']:.2f}\n" for _, r in data.iterrows()])
            return resp
        else: return f"No prediction data found for {found_stock}."

    return None

# Add this to your existing CSS in the main() function
st.markdown("""
<style>
/* Chatbot table formatting */
div.stChatMessage table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.9em;
}

div.stChatMessage th, div.stChatMessage td {
    padding: 8px;
    text-align: right;
    border: 1px solid #ddd;
}

div.stChatMessage th:first-child, div.stChatMessage td:first-child {
    text-align: left;
}

div.stChatMessage th {
    background-color: #f2f2f2;
    font-weight: bold;
}

div.stChatMessage tr:nth-child(even) {
    background-color: #f9f9f9;
}

div.stChatMessage tr:hover {
    background-color: #f1f1f1;
}
</style>
""", unsafe_allow_html=True)


# --- Data Fetching and Processing ---

@st.cache_data(ttl=900)
def get_stock_data(symbol, exchange, period="1y", interval="1d"):
    """Fetches stock historical data and info using yfinance."""
    formatted_symbol = symbol
    if exchange == "NSE": formatted_symbol = f"{symbol}.NS"
    elif exchange == "BSE": formatted_symbol = f"{symbol}.BO"

    try:
        stock = yf.Ticker(formatted_symbol)
        
        # First try to get history data
        hist = stock.history(period=period, interval=interval)
        
        # Initialize info with default values
        required_info = {
            "shortName": symbol, 
            "marketCap": None, 
            "trailingPE": None,
            "fiftyTwoWeekLow": None, 
            "fiftyTwoWeekHigh": None, 
            "sector": "N/A",
            "industry": "N/A", 
            "logo_url": None, 
            "currentPrice": None, 
            "previousClose": None
        }
        
        # Try to get stock info safely
        try:
            info = stock.info
            if info:
                for key in required_info:
                    if key in info and info[key] is not None:
                        required_info[key] = info[key]
        except (AttributeError, TypeError, KeyError) as e:
            st.warning(f"Could not retrieve detailed info for {formatted_symbol}: {str(e)}")
            # Continue with default values in required_info
        
        if hist.empty:
            st.warning(f"No historical data found for {formatted_symbol}. Check symbol/exchange.")
            # Attempt download as backup
            hist = yf.download(formatted_symbol, period=period, interval=interval, progress=False)
            if hist.empty:
                st.error(f"Backup download also failed for {formatted_symbol}. Cannot proceed.")
                return pd.DataFrame(), required_info

        # Try to get current price from hist if not in info
        if required_info.get('currentPrice') is None and not hist.empty:
            required_info['currentPrice'] = hist['Close'].iloc[-1]
        if required_info.get('previousClose') is None and len(hist) > 1:
            required_info['previousClose'] = hist['Close'].iloc[-2]

        return hist, required_info

    except Exception as e:
        st.error(f"Error fetching data for {formatted_symbol}: {e}")
        st.code(traceback.format_exc()) # Show traceback for fetch error
        return pd.DataFrame(), {
            "shortName": symbol, 
            "marketCap": None, 
            "trailingPE": None,
            "fiftyTwoWeekLow": None, 
            "fiftyTwoWeekHigh": None, 
            "sector": "N/A",
            "industry": "N/A", 
            "logo_url": None, 
            "currentPrice": None, 
            "previousClose": None
        }

# --- NEW: Robust News Scraping Helper ---
def scrape_yahoo_news(yahoo_url, symbol, company_name):
    """Attempts to scrape news from Yahoo Finance news page."""
    articles = []
    processed_urls = set()
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'Accept-Language': 'en-US,en;q=0.9'}
        response = requests.get(yahoo_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # ** Selector Strategy Update (Example - Needs Live Inspection!) **
        # Strategy 1: Find main list items directly (often using data-test or complex class)
        # news_items = soup.select('li[data-test="stream-root"] div div') # Example: This selector needs checking on live page
        # Strategy 2: Find any list and iterate through potential news containers
        list_items = soup.find_all('li')
        if not list_items:
            list_items = soup.find_all('div', class_=lambda x: x and ('news' in x or 'stream' in x or 'content' in x)) # Broader search


        count = 0
        for item in list_items:
             if count >= 5: break # Limit articles found via scrape
             title = None
             url = None
             source = "Yahoo Finance"
             published_date = datetime.now().strftime("%Y-%m-%d") # Default date

             # Try finding headline and link within the item
             link_tag = item.find('a', href=True)
             if link_tag:
                 # Try getting title from link's text or a heading inside it
                 headline_tag = link_tag.find(['h3', 'h2', 'div'], recursive=False) # Look for heading inside link
                 title = clean_text(headline_tag.get_text() if headline_tag else link_tag.get_text())
                 url = link_tag.get('href')

                 # Look for source/time info nearby (often sibling or parent divs)
                 meta_info_div = item.find('div', class_=lambda x: x and ('source' or 'provider' or 'meta' in x.lower()))
                 if meta_info_div:
                     source_text = clean_text(meta_info_div.get_text())
                     source_parts = source_text.split('•') # Common separator
                     if source_parts: source = source_parts[0].strip()
                     # Try to extract a better date if available (less common in list view)

                 # If title not found in link, check for a heading elsewhere in the item
                 if not title:
                      headline_tag = item.find(['h3', 'h2'])
                      if headline_tag: title = clean_text(headline_tag.get_text())


             # Construct absolute URL if relative
             if url and not url.startswith('http'):
                  parsed_uri = urllib.parse.urlparse(yahoo_url)
                  base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                  url = urllib.parse.urljoin(base_url, url)

             # Relevance check and add
             if title and url and len(title) > 15 and url not in processed_urls and (company_name.lower() in title.lower() or symbol.lower() in title.lower()):
                 articles.append({"title": title, "source": source, "url": url, "published": published_date})
                 processed_urls.add(url)
                 count += 1

       # if articles: st.info(f"Found {len(articles)} potential articles via Yahoo scraping.")
        return articles

    except requests.exceptions.RequestException as e:
        st.warning(f"Yahoo Finance news request failed ({yahoo_url}): {str(e)}")
    except Exception as e:
        st.warning(f"Error parsing Yahoo Finance news structure: {str(e)}")
    return [] # Return empty list on error

@st.cache_data(ttl=3600)
def get_company_news(symbol, exchange, company_name, news_api_key=None):
    """Fetch recent news articles using multiple methods."""
    all_articles = []
    processed_urls = set()
    max_articles_per_source = 3
    max_total_articles = 10

    # Determine Yahoo symbol
    yahoo_symbol = symbol
    if exchange == "NSE": yahoo_symbol = f"{symbol}.NS"
    elif exchange == "BSE": yahoo_symbol = f"{symbol}.BO"

    # --- Method 1: yfinance stock.news (often limited) ---
    try:
        stock = yf.Ticker(yahoo_symbol)
        yfinance_news = stock.news
        count = 0
        if yfinance_news:
             #st.info("Checking news via yfinance Ticker.news...")
             for item in yfinance_news[:max_articles_per_source]:
                 url = item.get('link')
                 title = item.get('title')
                 if url and title and url not in processed_urls:
                     publisher = item.get('publisher', 'yfinance')
                     pub_time_unix = item.get('providerPublishTime')
                     pub_date = datetime.fromtimestamp(pub_time_unix).strftime('%Y-%m-%d') if pub_time_unix else datetime.now().strftime("%Y-%m-%d")
                     all_articles.append({"title": title, "source": publisher, "url": url, "published": pub_date})
                     processed_urls.add(url)
                     count += 1
             #st.info(f"Found {count} articles via yfinance Ticker.news.")
    except Exception as e:
        st.warning(f"Could not fetch news via yfinance Ticker for {yahoo_symbol}: {e}")


    # --- Method 2: Scrape Yahoo Finance News Page (More Articles, Less Stable) ---
    if len(all_articles) < max_total_articles: # Only scrape if needed
        #st.info("Attempting scraping Yahoo Finance news page...")
        yahoo_url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/news"
        scraped_articles = scrape_yahoo_news(yahoo_url, symbol, company_name)
        for article in scraped_articles:
             if article['url'] not in processed_urls and len(all_articles) < max_total_articles:
                  all_articles.append(article)
                  processed_urls.add(article['url'])

    # --- Method 3: News API (if key provided and more needed) ---
    if news_api_key and len(all_articles) < max_total_articles:
        st.info("Fetching news from News API...")
        try:
            query = f'"{company_name}" OR "{symbol}" stock'
            news_api_url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&sortBy=relevancy&apiKey={news_api_key}&language=en&pageSize=10"
            response = requests.get(news_api_url, timeout=15)
            response.raise_for_status()
            news_data = response.json()
            if news_data.get('status') == 'ok' and news_data.get('articles'):
                count = 0
                for article in news_data['articles']:
                     url = article.get('url')
                     title = article.get('title')
                     # Basic check for relevance and avoid duplicates
                     if title and url and url not in processed_urls and (company_name.lower() in title.lower() or symbol.lower() in title.lower()):
                          if len(all_articles) >= max_total_articles: break
                          all_articles.append({
                              "title": clean_text(title), "source": article.get('source', {}).get('name', 'News API'),
                              "url": url, "published": article.get('publishedAt', '')[:10]
                          })
                          processed_urls.add(url)
                          count += 1
                st.info(f"Found {count} relevant articles via News API.")
            else: st.warning(f"News API returned status '{news_data.get('status')}' or no articles.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401: st.error("News API Key is invalid.")
            elif e.response.status_code == 429: st.warning("News API rate limit.")
            else: st.warning(f"News API HTTP error {e.response.status_code}: {e.response.text[:100]}...")
        except Exception as e: st.warning(f"Error processing News API: {str(e)}")
    elif not news_api_key: st.info("News API key not provided. Skipping.")


    # --- Final Processing ---
    unique_articles = []
    seen_titles = set()
    for article in all_articles:
        cleaned_title_lower = article['title'].lower()
        if cleaned_title_lower and cleaned_title_lower not in seen_titles:
            seen_titles.add(cleaned_title_lower)
            unique_articles.append(article)

    def relevance_score(article):
        title_lower = article['title'].lower()
        score = 0
        if company_name.lower() in title_lower: score += 3
        if symbol.lower() in title_lower: score += 2
        if 'stock' in title_lower or 'share' in title_lower: score += 1
        return score
    unique_articles.sort(key=relevance_score, reverse=True)

    return unique_articles[:max_total_articles]

# --- NEW: Robust Peer Scraping Helper ---
def scrape_yahoo_peers(yahoo_url, yahoo_symbol):
    """Attempts to scrape competitor symbols from Yahoo Finance profile page."""
    peer_symbols = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(yahoo_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple approaches to find peer companies
        
        # Approach 1: Look for specific peer section by common class names
        peer_sections = soup.find_all('section', class_=lambda x: x and ('peer' in x.lower() or 'competitor' in x.lower()))
        
        # Approach 2: Look for headers that might indicate peer sections
        if not peer_sections:
            headers = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r'Competitors|Peers|Similar Companies', re.I))
            for header in headers:
                parent = header.find_parent('section') or header.find_parent('div')
                if parent:
                    peer_sections.append(parent)
        
        # Approach 3: Try to find the peer table directly
        if not peer_sections:
            tables = soup.find_all('table')
            for table in tables:
                if table.find_previous(['h2', 'h3', 'h4'], string=re.compile(r'Competitors|Peers|Similar', re.I)):
                    peer_sections.append(table)
        
        # Approach 4: If we're on profile page, try to navigate to competitors page
        if not peer_sections:
            # Try to construct and fetch the competitors page URL
            competitors_url = yahoo_url.replace('/profile', '/competitors')
            comp_response = requests.get(competitors_url, headers=headers, timeout=15)
            if comp_response.status_code == 200:
                comp_soup = BeautifulSoup(comp_response.content, 'html.parser')
                peer_sections = comp_soup.find_all('table')
        
        # Extract peer symbols from any sections we found
        for section in peer_sections:
            links = section.find_all('a', href=re.compile(r'/quote/([^/?]+)'))
            for link in links:
                match = re.search(r'/quote/([^/?]+)', link.get('href', ''))
                if match:
                    peer_sym = match.group(1)
                    if peer_sym.lower() != yahoo_symbol.lower():  # Don't add self
                        peer_symbols.append(peer_sym)
        
        # Fallback: If all else fails, try to find any stock links on the page
        if not peer_symbols:
            all_stock_links = soup.find_all('a', href=re.compile(r'/quote/([^/?]+)'))
            for link in all_stock_links:
                match = re.search(r'/quote/([^/?]+)', link.get('href', ''))
                if match:
                    peer_sym = match.group(1)
                    # Filter out the current stock and common non-peer links
                    if (peer_sym.lower() != yahoo_symbol.lower() and 
                        not re.search(r'(index|chart|options|holders|financials|analysis|profile)', peer_sym.lower())):
                        peer_symbols.append(peer_sym)
        
        # Limit to first 10 unique peers if we found many
        peer_symbols = list(dict.fromkeys(peer_symbols))[:10]
        
        if peer_symbols:
            st.info(f"Found {len(peer_symbols)} potential peer symbols via scraping.")
        else:
            # If we still didn't find peers, use a hardcoded list of common peers for major exchanges
            exchange_suffix = yahoo_symbol.split('.')[-1] if '.' in yahoo_symbol else ''
            base_symbol = yahoo_symbol.split('.')[0]
            
            # For Indian stocks (NSE)
            if exchange_suffix == 'NS':
                if 'RELIANCE' in base_symbol:
                    peer_symbols = ['ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'HINDPETRO.NS']
                elif any(sector in base_symbol for sector in ['INFO', 'TCS', 'WIPRO', 'TECH']):
                    peer_symbols = ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS']
                elif any(bank in base_symbol for bank in ['SBI', 'HDFC', 'ICICI', 'AXIS', 'KOTAK']):
                    peer_symbols = ['SBIN.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']
            # For US stocks
            elif not exchange_suffix:
                if any(tech in base_symbol for tech in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']):
                    peer_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
            
            if peer_symbols:
                st.info(f"Using default peer list for {yahoo_symbol}: {', '.join(peer_symbols)}")
            
        return peer_symbols

    except requests.exceptions.RequestException as e:
        st.warning(f"Yahoo Finance peer request failed ({yahoo_url}): {str(e)}")
    except Exception as e:
        st.warning(f"Error parsing Yahoo Finance profile page for peers: {str(e)}")
    
    return peer_symbols


@st.cache_data(ttl=3600)
def get_peer_comparison(symbol, exchange, sector=None):
    """Fetches and processes peer comparison data."""
    peers_info = []
    max_peers_to_fetch = 5 # Limit how many peers we actually fetch data for

    # --- Step 1: Try scraping Yahoo Finance profile page ---
    yahoo_symbol = symbol
    if exchange == "NSE": yahoo_symbol = f"{symbol}.NS"
    elif exchange == "BSE": yahoo_symbol = f"{symbol}.BO"
    profile_url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/profile"

    ###st.info(f"Attempting to scrape peers from Yahoo profile: {profile_url}")
    scraped_peer_symbols = scrape_yahoo_peers(profile_url, yahoo_symbol)

    # --- Step 2: (Optional Fallback) Use Sector if scraping fails and sector known ---
    # if not scraped_peer_symbols and sector and sector != 'N/A':
    #    st.info(f"Scraping failed. Searching for stocks in sector '{sector}' (This is complex)...")
    #    # TODO: Implement sector-based search if desired (e.g., using a library/API or another scrape)
    #    pass

    # --- Step 3: Fetch data for identified peers ---
    final_peer_symbols = scraped_peer_symbols[:max_peers_to_fetch]

    if final_peer_symbols:
         st.info(f"Fetching detailed data for peers: {', '.join(final_peer_symbols)}")
         with concurrent.futures.ThreadPoolExecutor(max_workers=max_peers_to_fetch) as executor:
            future_to_peer = {}
            for peer_sym_original in final_peer_symbols:
                 # Simple exchange guessing logic (can be improved)
                 peer_exchange = "NASDAQ" # Default guess
                 if ".NS" in peer_sym_original: peer_exchange = "NSE"
                 elif ".BO" in peer_sym_original: peer_exchange = "BSE"
                 elif "." not in peer_sym_original: pass # Keep NASDAQ/NYSE guess
                 else: peer_exchange = "OTHER" # Or use yfinance default

                 clean_peer_symbol = peer_sym_original.split('.')[0]
                 # Submit task to fetch data
                 future = executor.submit(get_stock_data, clean_peer_symbol, peer_exchange, period="3mo") # Fetch more history
                 future_to_peer[future] = peer_sym_original

            for future in concurrent.futures.as_completed(future_to_peer):
                peer_sym_original = future_to_peer[future]
                try:
                    hist, peer_info = future.result()
                    if not hist.empty and peer_info:
                        # Robust price finding
                        current_price = peer_info.get('currentPrice') or peer_info.get('previousClose')
                        if current_price is None and not hist.empty:
                            current_price = hist['Close'].iloc[-1]

                        # Robust 1-month change calc
                        change_1m = 'N/A'
                        if current_price is not None and len(hist) > 20:
                            try:
                                one_month_ago = hist.index[-1] - pd.Timedelta(days=30)
                                closest_date_index = hist.index.get_indexer([one_month_ago], method='nearest')[0]
                                price_1m_ago = hist['Close'].iloc[closest_date_index]
                                if price_1m_ago is not None and isinstance(price_1m_ago, (int,float)) and abs(price_1m_ago) > 1e-6:
                                     change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
                            except Exception as calc_err:
                                 # st.warning(f"Could not calc 1M change for {peer_sym_original}: {calc_err}")
                                 pass # Keep N/A

                        if current_price is not None: # Only add peer if we have a current price
                            peers_info.append({
                                "Symbol": peer_sym_original.split('.')[0],
                                "Name": peer_info.get('shortName', peer_sym_original),
                                "Current Price": current_price,
                                "Change (1M)": change_1m,
                                "Market Cap": peer_info.get('marketCap'),
                                "P/E Ratio": peer_info.get('trailingPE')
                            })
                    # else: st.warning(f"No data returned for peer {peer_sym_original}")

                except Exception as peer_err:
                    st.warning(f"Error processing peer {peer_sym_original}: {peer_err}")
                    # st.code(traceback.format_exc()) # Enable for deep debug
                    continue
    else:
        st.info("No peer symbols identified to fetch data for.")

    return peers_info

# --- Plotting Functions ---

def plot_stock_price_chart(hist_data, company_name, exchange):
    """Creates a Plotly chart with candlestick, volume, and moving averages."""
    if hist_data.empty: return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=(f'{company_name} Price', 'Volume'),
                        row_heights=[0.7, 0.3])
    try: # Add top-level try for chart generation
         fig.add_trace(go.Candlestick(x=hist_data.index,
                                      open=hist_data['Open'], high=hist_data['High'],
                                      low=hist_data['Low'], close=hist_data['Close'], name='Price'), row=1, col=1)
         if len(hist_data) >= 20:
             hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
             fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA20'], mode='lines', name='MA 20', line=dict(color='orange', width=1)), row=1, col=1)
         if len(hist_data) >= 50:
             hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
             fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA50'], mode='lines', name='MA 50', line=dict(color='purple', width=1)), row=1, col=1)
         fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

         chart_title = f'{company_name} ({hist_data.index.min().strftime("%Y-%m-%d")} to {hist_data.index.max().strftime("%Y-%m-%d")})'
         fig.update_layout(title=chart_title, xaxis_rangeslider_visible=False, hovermode='x unified',
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
         fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
         fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
         currency_symbol = "₹" if exchange in ['NSE', 'BSE'] else "$"
         fig.update_yaxes(row=1, col=1, title_text=f"Price ({currency_symbol})")
         fig.update_yaxes(row=2, col=1, title_text="Volume")
         return fig
    except Exception as plot_err:
        st.error(f"Error generating price chart: {plot_err}")
        return None


def plot_sentiment_analysis(sentiment_data):
    """Creates charts for sentiment analysis results."""
    if not sentiment_data or not sentiment_data.get('sentiment_counts'): return None
    try: # Add try-except for robustness
         counts = sentiment_data['sentiment_counts']
         labels = list(counts.keys())
         values = list(counts.values())
         colors = {'Positive': '#10B981', 'Neutral': '#F59E0B', 'Negative': '#EF4444'}
         marker_colors = [colors.get(label, '#CCCCCC') for label in labels]

         fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'xy'}]],
                             subplot_titles=('Overall Score', 'Headline Distribution'))

         overall_sentiment_label = sentiment_data.get('overall_sentiment', 'neutral').capitalize()
         gauge_color = colors.get(overall_sentiment_label, '#F59E0B')

         fig.add_trace(go.Indicator(
             mode="gauge+number", value=sentiment_data.get('sentiment_score', 5),
             domain={'x': [0, 1], 'y': [0, 1]}, title={'text': f"Overall: {overall_sentiment_label}", 'font': {'size': 16}},
             gauge={'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': gauge_color},
                    'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "gray",
                    'steps': [{'range': [1, 4], 'color': '#FEE2E2'}, {'range': [4, 7], 'color': '#FEF3C7'}, {'range': [7, 10], 'color': '#D1FAE5'}]}), row=1, col=1)
         fig.add_trace(go.Bar(x=labels, y=values, marker_color=marker_colors, name='Counts'), row=1, col=2)
         fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
         fig.update_yaxes(title_text="Headlines", row=1, col=2, title_font_size=12, tickfont_size=10)
         fig.update_xaxes(tickfont_size=10, row=1, col=2)
         return fig
    except Exception as e:
        st.error(f"Error generating sentiment plot: {e}")
        return None


def plot_peer_comparison(peer_data, main_symbol, main_price, exchange):
    """Creates bar charts comparing the main stock with its peers."""
    if not peer_data: return None, None
    try: main_price_numeric = float(main_price) if main_price is not None else None
    except (ValueError, TypeError): main_price_numeric = None

    comparison_list = peer_data + [{"Symbol": main_symbol, "Name": f"{main_symbol} (You)", "Current Price": main_price_numeric, "Change (1M)": "N/A"}]
    df = pd.DataFrame(comparison_list)
    df_price = df.dropna(subset=['Current Price', 'Name']).copy()
    currency_symbol = "₹" if exchange in ['NSE', 'BSE'] else "$"

    fig_price, fig_perf = None, None # Initialize plots

    try: # Price Chart
        if not df_price.empty:
             df_price = df_price.sort_values("Current Price", ascending=False)
             fig_price = px.bar(df_price, x='Name', y='Current Price', title='Peer Comparison: Current Price',
                                text='Current Price', color='Name', labels={'Name': 'Company', 'Current Price': f'Price ({currency_symbol})'})
             fig_price.update_traces(texttemplate=f'{currency_symbol}%{{text:,.2f}}', textposition='outside') # Add comma formatting
             fig_price.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False, height=400)
        else: fig_price = _create_empty_plot('Peer Comparison: Current Price', 'No valid price data for comparison.', height=400)
    except Exception as e:
        st.error(f"Error generating peer price chart: {e}")
        fig_price = _create_empty_plot('Peer Comparison: Current Price', 'Error generating chart.', height=400)


    try: # Performance Chart
        df_perf = df.copy()
        df_perf['Change (1M)'] = pd.to_numeric(df_perf['Change (1M)'], errors='coerce')
        df_perf = df_perf.dropna(subset=['Change (1M)', 'Name'])
        if not df_perf.empty:
            df_perf = df_perf.sort_values("Change (1M)", ascending=False)
            fig_perf = px.bar(df_perf, x='Name', y='Change (1M)', title='Peer Comparison: 1-Month Price Change (%)',
                              text='Change (1M)', color='Change (1M)', color_continuous_scale=px.colors.diverging.RdYlGn,
                              labels={'Name': 'Company', 'Change (1M)': 'Change (%)'})
            fig_perf.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_perf.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', coloraxis_showscale=False, showlegend=False, height=400)
        else: fig_perf = _create_empty_plot('Peer Comparison: 1-Month Price Change (%)', 'No 1-Month Change data for comparison.', height=400)
    except Exception as e:
         st.error(f"Error generating peer performance chart: {e}")
         fig_perf = _create_empty_plot('Peer Comparison: 1-Month Price Change (%)', 'Error generating chart.', height=400)

    return fig_price, fig_perf

def _create_empty_plot(title, message, height=300):
     """Helper to create a placeholder plot."""
     fig = go.Figure()
     fig.update_layout(title=title, height=height, xaxis={'visible': False}, yaxis={'visible': False},
                       annotations=[{'text': message, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 14}}])
     return fig

def email_subscription_form():
    st.subheader("Subscribe to Hourly Stock Updates")
    
    with st.form(key="email_subscription"):
        email = st.text_input("Your Email Address", placeholder="example@domain.com")
        stock_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, MSFT, RELIANCE")
        exchange = st.selectbox("Exchange", options=["NSE", "BSE", "NASDAQ"], index=0)
        
        submit_button = st.form_submit_button(label="Subscribe")
        
        if submit_button:
            if not email or "@" not in email or "." not in email:
                st.error("Please enter a valid email address.")
                return None
            
            if not stock_symbol:
                st.error("Please enter a stock symbol.")
                return None
            
            # Store subscription in session state
            if 'email_subscriptions' not in st.session_state:
                st.session_state.email_subscriptions = []
            
            # Check if already subscribed
            for sub in st.session_state.email_subscriptions:
                if sub['email'] == email and sub['stock'] == stock_symbol and sub['exchange'] == exchange:
                    st.warning("You're already subscribed for this stock.")
                    return None
            
            # Add new subscription
            st.session_state.email_subscriptions.append({
                'email': email,
                'stock': stock_symbol,
                'exchange': exchange,
                'last_sent': None  # Track when the last email was sent
            })
            
            st.success(f"Successfully subscribed to hourly updates for {stock_symbol} on {exchange}!")
            return {'email': email, 'stock': stock_symbol, 'exchange': exchange}
    
    return None

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time

def send_stock_update_email(recipient_email, stock_symbol, exchange, sender_email, app_password):
    """Send an email with stock sentiment analysis and news summary."""
    try:
        # Get stock data, news, and sentiment analysis
        hist_data, stock_info = get_stock_data(stock_symbol, exchange, period="1mo")
        news_articles = get_company_news(stock_symbol, exchange, stock_info.get('shortName', stock_symbol))
        
        # Perform sentiment analysis on news headlines if we have news
        sentiment_data = None
        if news_articles and len(news_articles) > 0:
            headlines = [article['title'] for article in news_articles if 'title' in article]
            if headlines and st.session_state.sentiment_analyzer:
                sentiment_data = st.session_state.sentiment_analyzer.analyze_sentiment(headlines)
        
        # Create email content
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Stock Update: {stock_symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Email body
        email_body = f"""
        <html>
        <body>
        <h2>Stock Update: {stock_symbol} ({exchange})</h2>
        <p><b>Current Price:</b> {format_large_number(stock_info.get('currentPrice'), currency=True, exchange=exchange)}</p>
        """
        
        # Add sentiment analysis if available
        if sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score', 'N/A')
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral').capitalize()
            email_body += f"""
            <h3>Sentiment Analysis</h3>
            <p><b>Overall Sentiment:</b> {overall_sentiment}</p>
            <p><b>Sentiment Score:</b> {sentiment_score}/10</p>
            """
            
            # Add sentiment counts
            if 'sentiment_counts' in sentiment_data:
                email_body += "<p><b>Sentiment Distribution:</b></p><ul>"
                for sentiment, count in sentiment_data['sentiment_counts'].items():
                    email_body += f"<li>{sentiment}: {count}</li>"
                email_body += "</ul>"
        
        # Add recent news
        if news_articles and len(news_articles) > 0:
            email_body += "<h3>Recent News</h3><ul>"
            for i, article in enumerate(news_articles[:5]):  # Limit to 5 news items
                email_body += f"<li><b>{article['title']}</b> - {article.get('source', 'Unknown')}</li>"
            email_body += "</ul>"
        
        email_body += """
        <p>This is an automated email from the Stock Analysis Assistant.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(email_body, 'html'))
        
        # Connect to Gmail's SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        
        return True, "Email sent successfully!"
    
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

import threading
import schedule
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Global variable to track if the scheduler is running
scheduler_running = False

def initialize_email_scheduler(sender_email, app_password):
    """Initialize the email scheduler to run in the background."""
    global scheduler_running
    
    if scheduler_running:
        return  # Scheduler already running
    
    def send_scheduled_emails():
        """Function to send emails to all subscribers."""
        if 'email_subscriptions' not in st.session_state:
            return
        
        current_time = datetime.now()
        
        for subscription in st.session_state.email_subscriptions:
            # Check if it's been at least a minute since the last email (changed from hour)
            if (subscription['last_sent'] is None or 
                (current_time - subscription['last_sent']).total_seconds() >= 60):  # 60 seconds instead of 3600
                
                success, message = send_stock_update_email(
                    subscription['email'],
                    subscription['stock'],
                    subscription['exchange'],
                    sender_email,
                    app_password
                )
                
                if success:
                    subscription['last_sent'] = current_time
                    print(f"Sent update for {subscription['stock']} to {subscription['email']}")
                else:
                    print(f"Failed to send email: {message}")
    
    def run_scheduler():
        """Run the scheduler in a loop."""
        global scheduler_running
        scheduler_running = True
        
        # Schedule the task to run every minute instead of hour
        schedule.every(1).minutes.do(send_scheduled_emails)
        
        while True:
            schedule.run_pending()
            time.sleep(1)  # Check every second if there are pending tasks
    
    # Get the current ScriptRunContext
    ctx = get_script_run_ctx()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    
    # Add the ScriptRunContext to the thread
    add_script_run_ctx(scheduler_thread, ctx)
    
    scheduler_thread.start()
    
    st.session_state.scheduler_initialized = True


def test_email_functionality():
    """A function to test the email sending functionality."""
    st.subheader("Test Email Functionality")
    
    with st.form(key="test_email"):
        test_email = st.text_input("Test Email Address", placeholder="example@domain.com")
        test_stock = st.text_input("Test Stock Symbol", value="AAPL")
        test_exchange = st.selectbox("Test Exchange", options=["NSE", "BSE", "NASDAQ"], index=2)
        
        test_button = st.form_submit_button(label="Send Test Email")
        
        if test_button:
            if not test_email or "@" not in test_email or "." not in test_email:
                st.error("Please enter a valid email address.")
                return
            
            sender_email = "126179033@sastra.ac.in"
            app_password = "qgco imng qurh ehwl"
            
            with st.spinner("Sending test email..."):
                success, message = send_stock_update_email(
                    test_email, 
                    test_stock, 
                    test_exchange, 
                    sender_email, 
                    app_password
                )
                
                if success:
                    st.success(f"Test email sent successfully to {test_email}!")
                else:
                    st.error(f"Failed to send test email: {message}")

# --- Main Application ---

def main():
    # Custom CSS for styling (Keep as before or adjust)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keep your existing CSS here

    st.markdown('<h1 class="main-header">Advanced Stock Analysis Assitant</h1>', unsafe_allow_html=True)

    # --- Initialization ---
    # Minimize console info logs during startup
    if 'app_initialized' not in st.session_state:
        st.info("Initializing application components...")
        # Sentiment Analyzer
        if ensure_model_is_cached('ProsusAI/finbert'):
            st.session_state.sentiment_analyzer = FinancialNLP()
        else: st.error("FinBERT model unavailable. Sentiment analysis disabled.")
        # GPT-2 Model
        model, tokenizer, device = load_gpt2_model()
        if model and tokenizer:
            st.session_state.gpt2_model, st.session_state.gpt2_tokenizer = model, tokenizer
        else: st.warning("GPT-2 model unavailable. Chatbot disabled.")
        # Predictions Data
        st.session_state.stock_predictions = load_stock_predictions()
        st.session_state.app_initialized = True # Mark as initialized
        st.info("Initialization complete.")

        # Add sender email and app password
    sender_email = "126179033@sastra.ac.in"
    app_password = "qgco imng qurh ehwl"
    
    # Initialize the email scheduler if not already running
    if 'scheduler_initialized' not in st.session_state:
        initialize_email_scheduler(sender_email, app_password)
    
    # Add a sidebar section for email subscription
    st.sidebar.markdown("---")
    with st.sidebar.expander("📧 Subscribe to Stock Updates", expanded=False):
        email_subscription_form()
        
        # Show current subscriptions
        if 'email_subscriptions' in st.session_state and st.session_state.email_subscriptions:
            st.write("Your Current Subscriptions:")
            for sub in st.session_state.email_subscriptions:
                st.write(f"• {sub['stock']} ({sub['exchange']}) - {sub['email']}")
                
                # Add an unsubscribe button
                if st.button(f"Unsubscribe from {sub['stock']}", key=f"unsub_{sub['stock']}_{sub['email']}"):
                    st.session_state.email_subscriptions.remove(sub)
                    st.experimental_rerun()
    
    # Add a tab for testing email functionality
    tabs = st.tabs(["Stock Analysis", "News & Sentiment", "Peer Comparison", "Email Testing"])
    
    with tabs[3]:  # Email Testing tab
        test_email_functionality()


    # --- Sidebar ---
    with st.sidebar:
        # Use use_container_width
        st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg", use_container_width=True)
        st.markdown('<h2 class="subheader" style="margin-top: 0.5rem; border-bottom: none;">Stock Selection</h2>', unsafe_allow_html=True)

        # Inputs using session state for history persistence
        if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = "RELIANCE"
        if 'selected_exchange' not in st.session_state: st.session_state.selected_exchange = "NSE"
        if 'selected_company_name' not in st.session_state: st.session_state.selected_company_name = "Reliance Industries"

        symbol = st.text_input("Stock Symbol", st.session_state.selected_symbol, key="input_symbol", help="E.g., RELIANCE, AAPL, INFY").upper()
        exchange_options = ["NSE", "BSE", "NASDAQ", "NYSE"]
        exchange_index = exchange_options.index(st.session_state.selected_exchange) if st.session_state.selected_exchange in exchange_options else 0
        exchange = st.selectbox("Exchange", exchange_options, index=exchange_index, key="input_exchange", help="Select the stock exchange.")

        default_name = symbol # Simple default - add more mappings if needed
        # Use saved name from history/previous analysis if available
        company_name_value = st.session_state.selected_company_name if st.session_state.selected_company_name else default_name
        company_name = st.text_input("Company Name (for news)", company_name_value, key="input_company_name", help="Used for news search relevance.")

        st.markdown("---")
        st.markdown("**Chart Settings**")
        period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
        interval = st.selectbox("Data Interval", ["1d", "5d", "1wk", "1mo"], index=0)
        st.markdown("---")
        news_api_key = st.text_input("News API Key (optional)", type="password", help="Get one from newsapi.org for better news results.")
        search_button = st.button("Analyze Stock", type="primary", use_container_width=True)

        # Chatbot
        st.markdown('<h2 class="subheader" style="border-bottom: none;">Chatbot</h2>', unsafe_allow_html=True)
        st.caption("Ask about loaded prediction data.")
        chat_query = st.text_input("Ask:", placeholder="e.g., Top 5 stocks next 7 days?")
        chat_button = st.button("Ask Chatbot")
        if chat_button and chat_query:
             if st.session_state.get('gpt2_model') and st.session_state.get('gpt2_tokenizer'):
                 with st.spinner("🤖 Thinking..."):
                     response = generate_chatbot_response(chat_query)
                     st.markdown(f"**Q:** {chat_query}\n\n**A:** {response}")
             else: st.warning("Chatbot model is unavailable.")

        # Search History
        st.markdown('<h2 class="subheader" style="border-bottom: none;">History</h2>', unsafe_allow_html=True)
        if st.session_state.search_history:
            for i, item in enumerate(reversed(st.session_state.search_history)): # Show latest first
                hist_label = f"{item['symbol']} ({item['exchange']}) {item['date']}"
                button_key = f"history_{item['symbol']}_{item['exchange']}_{i}"
                if st.button(hist_label, key=button_key, help=f"Reload {item['symbol']}", use_container_width=True):
                    st.session_state.selected_symbol = item['symbol']
                    st.session_state.selected_exchange = item['exchange']
                    st.session_state.selected_company_name = item['company_name']
                    st.session_state.history_clicked = True
                    st.rerun()


    # --- Main Content Area ---
    if st.session_state.get('history_clicked', False):
        st.info(f"Loaded '{st.session_state.selected_symbol}' from history. Click 'Analyze Stock' to run analysis.")
        st.session_state.history_clicked = False # Reset flag

    if search_button:
        # Update session state from current inputs before analysis
        st.session_state.selected_symbol = symbol
        st.session_state.selected_exchange = exchange
        st.session_state.selected_company_name = company_name

        # Add to search history (limit size)
        history_entry = {'symbol': symbol, 'exchange': exchange, 'company_name': company_name, 'date': datetime.now().strftime("%H:%M")}
        if not st.session_state.search_history or (st.session_state.search_history[-1]['symbol'] != symbol or st.session_state.search_history[-1]['exchange'] != exchange):
            st.session_state.search_history.append(history_entry)
            if len(st.session_state.search_history) > 5: st.session_state.search_history.pop(0)

        st.session_state.last_analyzed = history_entry # Store what's being analyzed

        with st.spinner(f"Analyzing {symbol} ({exchange})..."):
             # --- Fetch Data ---
             hist_data, info = get_stock_data(symbol, exchange, period, interval)

             if hist_data.empty:
                 st.error(f"Failed to fetch data for {symbol} ({exchange}). Analysis halted.")
             else:
                 # --- Overview ---
                 st.markdown(f'<h2 class="subheader">Overview: {info.get("shortName", company_name)} ({symbol})</h2>', unsafe_allow_html=True)
                 col1, col2, col3 = st.columns(3)
                 current_price = hist_data['Close'].iloc[-1] if not hist_data.empty else 'N/A'
                 with col1:
                    # Name
                    st.markdown('<div class="metric-card"><div class="metric-label">Company Name</div><div class="metric-value">{}</div></div>'.format(info.get("shortName", company_name)), unsafe_allow_html=True)
                    # Price
                    price_str = format_large_number(current_price, currency=True, exchange=exchange)
                    st.markdown('<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">{}</div></div>'.format(price_str), unsafe_allow_html=True)
                 with col2:
                    # Market Cap
                    mcap_str = format_large_number(info.get('marketCap'), currency=True, exchange=exchange)
                    st.markdown('<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{}</div></div>'.format(mcap_str), unsafe_allow_html=True)
                    # P/E Ratio
                    pe_str = f"{info.get('trailingPE'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A'
                    st.markdown('<div class="metric-card"><div class="metric-label">P/E Ratio (TTM)</div><div class="metric-value">{}</div></div>'.format(pe_str), unsafe_allow_html=True)
                 with col3:
                     # 52 Week Range
                     wk_low, wk_high = info.get('fiftyTwoWeekLow'), info.get('fiftyTwoWeekHigh')
                     range_str = "N/A"
                     if isinstance(wk_low, (int,float)) and isinstance(wk_high, (int,float)):
                         low_str = format_large_number(wk_low, False); high_str = format_large_number(wk_high, False)
                         curr_sym = "₹" if exchange in ['NSE','BSE'] else "$"
                         range_str = f'<div class="metric-value" style="font-size: 1.1rem;">{curr_sym}{low_str} - {curr_sym}{high_str}</div>'
                     else: range_str = '<div class="metric-value">N/A</div>'
                     st.markdown(f'<div class="metric-card"><div class="metric-label">52 Week Range</div>{range_str}</div>', unsafe_allow_html=True)
                     # Day Change
                     day_change, day_change_class, day_change_str = 0.0, "neutral", "N/A"
                     if len(hist_data) > 1:
                         prev_close = hist_data['Close'].iloc[-2]
                         if isinstance(prev_close,(int,float)) and prev_close!=0 and isinstance(current_price,(int,float)):
                              day_change = ((current_price/prev_close)-1)*100; day_change_class = "positive" if day_change>=0 else "negative"
                              day_change_str = f"{'+' if day_change>=0 else ''}{day_change:.2f}%"
                     st.markdown('<div class="metric-card"><div class="metric-label">Day Change</div><div class="metric-value {}">{}</div></div>'.format(day_change_class, day_change_str), unsafe_allow_html=True)


                 # --- Price Chart ---
                 st.markdown('<h2 class="subheader">Price Chart & Technical Indicators</h2>', unsafe_allow_html=True)
                 price_chart_fig = plot_stock_price_chart(hist_data, info.get("shortName", company_name), exchange)
                 if price_chart_fig:
                     st.plotly_chart(price_chart_fig, use_container_width=True)
                 else: st.warning("Could not generate price chart.")

                 # --- News & Sentiment ---
                 st.markdown('<h2 class="subheader">Recent News & Sentiment</h2>', unsafe_allow_html=True)
                 with st.spinner("Fetching & analyzing news..."):
                      news = get_company_news(symbol, exchange, company_name, news_api_key)
                      if news:
                           headlines = [a['title'] for a in news]
                           sentiment_data = None
                           if st.session_state.get('sentiment_analyzer') and st.session_state.sentiment_analyzer.model:
                                sentiment_data = st.session_state.sentiment_analyzer.analyze_sentiment(headlines)
                           else: st.warning("Sentiment analysis unavailable.")

                           if sentiment_data:
                                sentiment_fig = plot_sentiment_analysis(sentiment_data)
                                if sentiment_fig: st.plotly_chart(sentiment_fig, use_container_width=True)
                           st.markdown("---") # Separator
                           for article in news:
                                sentiment, sentiment_class = "neutral", "sentiment-neutral"
                                if sentiment_data and "headline_sentiments" in sentiment_data:
                                    headline_map = dict(sentiment_data["headline_sentiments"]) # Easier lookup
                                    sentiment = headline_map.get(article['title'], "neutral")
                                    sentiment_class = f"sentiment-{sentiment}"

                                st.markdown(f"""
                                <div class="news-card">
                                    <span class="news-sentiment {sentiment_class}">{sentiment.capitalize()}</span>
                                    <div class="news-title">{article["title"]}</div>
                                    <div><span class="news-source">{article["source"]}</span> <span class="news-date">• {article["published"]}</span></div>
                                    <a href="{article["url"]}" target="_blank" class="news-link">Read Article 🔗</a>
                                </div>""", unsafe_allow_html=True)
                      else:
                           st.info("No recent news found matching the criteria.")

                 # --- Peer Comparison ---
                 st.markdown('<h2 class="subheader">Peer Comparison</h2>', unsafe_allow_html=True)
                 with st.spinner("Fetching peer comparison data..."):
                      peer_data = get_peer_comparison(symbol, exchange, info.get('sector'))
                      if peer_data:
                           price_for_plot = current_price if isinstance(current_price, (int, float)) else None
                           price_fig, perf_fig = plot_peer_comparison(peer_data, symbol, price_for_plot, exchange)
                           if price_fig and perf_fig:
                                col1, col2 = st.columns(2)
                                with col1: st.plotly_chart(price_fig, use_container_width=True)
                                with col2: st.plotly_chart(perf_fig, use_container_width=True)
                           elif price_fig: st.plotly_chart(price_fig, use_container_width=True)
                           elif perf_fig: st.plotly_chart(perf_fig, use_container_width=True)

                           st.markdown('<h3 style="font-size: 1.2rem; margin-top: 1rem; color: #1F2937;">Peer Data</h3>', unsafe_allow_html=True)
                           display_data = []
                           for co in peer_data:
                               chg_str = f"{co['Change (1M)']:.2f}%" if isinstance(co['Change (1M)'],(int,float)) else 'N/A'
                               pe_str = f"{co['P/E Ratio']:.2f}" if isinstance(co['P/E Ratio'],(int,float)) else 'N/A'
                               display_data.append({
                                   "Symbol": co["Symbol"], "Name": co["Name"],
                                   "Price": format_large_number(co["Current Price"], True, exchange),
                                   "Change(1M)": chg_str,
                                   "MarketCap": format_large_number(co["Market Cap"], True, exchange),
                                   "P/E Ratio": pe_str })
                           st.dataframe(pd.DataFrame(display_data).set_index('Symbol'), use_container_width=True)
                      else:
                           st.info("Could not retrieve peer comparison data.")


    # --- Welcome Message ---
    elif not st.session_state.last_analyzed and not st.session_state.get('history_clicked', False):
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; background-color: #F9FAFB; border-radius: 0.5rem; border: 1px solid #E5E7EB;">
            <h2 style="color: #1E3A8A;">Welcome!</h2>
            <p style="color: #374151; font-size: 1.1rem;">Enter stock details in the sidebar to begin analysis.</p>
            <p style="color: #4B5563;">Features:</p>
            <ul style="list-style-type: '➔ '; padding-left: 2rem; text-align: left; display: inline-block; color: #4B5563;">
                <li>Price Charts & Indicators</li>
                <li>News & Sentiment Analysis</li>
                <li>Peer Company Comparison</li>
                <li>Prediction Chatbot (if data loaded)</li>
            </ul>
        </div>""", unsafe_allow_html=True)


# --- Run the application ---
if __name__ == "__main__":
    # Fix for potential asyncio event loop issues in Streamlit/notebooks
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            st.warning(f"Unhandled Runtime Error during asyncio setup: {ex}")


    # --- Run the main Streamlit app function ---
    try:
        main()
    except Exception as main_err:
         st.error(f"An error occurred during application execution: {main_err}")
         st.error("Traceback:")
         st.code(traceback.format_exc())

# --- END OF FILE ft_llm.py ---