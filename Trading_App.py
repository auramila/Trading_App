import os
import json
import io
import requests
import random
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import firebase_admin
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# NLP
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Firebase
from firebase_admin import credentials, auth

# LangChain Chat Model
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# PDF export (for chat history)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Date/Time
from datetime import datetime, timedelta

# ------------------------------------------------
# SECURITY GUARD #1: Simple disallowed-content check
# ------------------------------------------------
# This is a basic text check for certain disallowed phrases or known jailbreak attempts.
# In production, you can add more robust checks or integrate an external Moderation API.
DISALLOWED_PHRASES = [
    "jailbreak",
    "DAN prompt",
    "hate speech",
    "racist",
    "violence"
]

def contains_disallowed_content(user_input: str) -> bool:
    input_lower = user_input.lower()
    return any(phrase in input_lower for phrase in DISALLOWED_PHRASES)

# -----------------------------------------------
# SECURITY GUARD #2: Rate/length limit
# -----------------------------------------------
# This helps mitigate extremely large or frequent requests that might break or spam the system.
LAST_REQUEST_TIME_KEY = "last_request_time"
REQUEST_COOLDOWN_SECONDS = 2  # minimal cooldown to prevent spamming
MAX_PROMPT_LENGTH = 1500      # maximum number of characters for a single prompt

def is_too_frequent_or_long(user_input: str) -> bool:
    current_time = time.time()
    last_time = st.session_state.get(LAST_REQUEST_TIME_KEY, 0)
    if current_time - last_time < REQUEST_COOLDOWN_SECONDS:
        return True
    if len(user_input) > MAX_PROMPT_LENGTH:
        return True
    st.session_state[LAST_REQUEST_TIME_KEY] = current_time
    return False

# ---------------------------
# Session State Initialization
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "user" not in st.session_state:
    st.session_state["user"] = None
if "user_csv_db" not in st.session_state:
    st.session_state["user_csv_db"] = None

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(
    page_title="StockTalk: Your Investment Buddy",
    layout="wide"
)

# ---------------------------
# App Introduction
# ---------------------------
st.title("🤖 StockTalk: Your Investment Buddy 🤑")
st.markdown("""
Welcome to **StockTalk** – your fun, simple way to explore stocks!

- **500 Most Talked Stocks:** We use a list of the 500 most buzzed-about stocks.
- **Market Snapshot:** See how major indices are doing today.
- **High Volatility Stocks:** Check out the top 10 stocks with the biggest weekly moves, along with their daily moves.
- **Chat & Analysis:** Ask any investment question. If your stock is on our list, you’ll also get a TradingView chart link!
- **Extra Ideas:** Not in our list? We can help with other investment insights too.

Let's dive in and have some fun with investing!
""")

# ---------------------------
# OpenAI and App Setup (Expandable)
# ---------------------------
with st.sidebar.expander("App Setup"):
    st.subheader("Login / Register")
    if st.session_state.get("user"):
        st.success(f"✅ Logged in as {st.session_state['user']}")
        if st.button("Logout", key="logout"):
            st.session_state["user"] = None
            st.info("You have been logged out. Refresh the page!")
    else:
        auth_mode = st.radio("Mode", ["Login", "Register"], key="auth_mode")
        auth_email = st.text_input("Email", key="auth_email")
        auth_password = st.text_input("Password", type="password", key="auth_password")
        if auth_mode == "Login":
            if st.button("Login", key="login"):
                try:
                    user = auth.get_user_by_email(auth_email)
                    st.session_state["user"] = user.email
                    st.success(f"Logged in as {user.email}")
                except Exception:
                    st.error("Login failed: Check your credentials.")
        else:
            if st.button("Register", key="register"):
                try:
                    user = auth.create_user(email=auth_email, password=auth_password)
                    st.success("Registration successful! Please log in.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")

    st.subheader("OpenAI Settings")
    openai_api_key_input = st.text_input("OpenAI API Key", type="password", key="openai_key")
    openai_model = st.selectbox("Model Name", ["gpt-4", "gpt-3.5-turbo"], key="model_name")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature")
    top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1, key="top_p")
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1, key="freq_penalty")
    presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1, key="pres_penalty")

# ---------------------------
# Load Other Environment Variables
# ---------------------------
load_dotenv()
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ---------------------------
# Initialize Firebase
# ---------------------------
try:
    if not firebase_admin._apps:
        cred_path = "firebase_credentials.json"  # Provide your Firebase credentials
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.sidebar.error(f"Firebase initialization error: {e}")

# ---------------------------
# Download NLTK Data
# ---------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ---------------------------
# Load Default CSV (500 Most Talked Stocks) if not loaded
# ---------------------------
if st.session_state.get("user_csv_db") is None:
    try:
        default_df = pd.read_csv("top500buzz.csv")
        default_df.columns = [col.lower() for col in default_df.columns]
        st.session_state["user_csv_db"] = default_df
    except Exception as e:
        st.sidebar.error(f"Error loading default CSV: {e}")

# ---------------------------
# Create OpenAI LLM Instance (Always OpenAI)
# ---------------------------
def create_llm():
    if openai_api_key_input:
        return ChatOpenAI(
            temperature=temperature,
            model_name=openai_model,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            openai_api_key=openai_api_key_input
        )
    else:
        st.warning("Please enter your OpenAI API key.")
        return None

llm = create_llm()

# ---------------------------
# Conversation Chain
# ---------------------------
conversation = None
if llm:
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)

# ---------------------------
# Cached Yahoo Finance Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def get_sector_for_ticker(ticker_symbol: str) -> str:
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.info.get("sector", "N/A")
    except Exception:
        return "N/A"

@st.cache_data(show_spinner=False)
def download_stock_history(ticker_value: str, period="3mo", interval="1d"):
    try:
        return yf.download(ticker_value, period=period, interval=interval, progress=False)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_index_history(ticker: str, period="2d"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_stock_info_cached(ticker_symbol: str):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        company_name = info.get("longName", "N/A")
        if company_name == "N/A":
            company_name = info.get("name", "N/A")
        officers = info.get("companyOfficers", [])
        ceo_name = "N/A"
        if officers and isinstance(officers, list) and len(officers) > 0:
            ceo_name = officers[0].get("name", "N/A")
        ceo_name = clean_ceo_name(ceo_name, company_name)
        forward_pe = info.get("forwardPE", 0)
        if not forward_pe:
            growth_outlook = "Unavailable"
        elif forward_pe > 20:
            growth_outlook = "High growth potential"
        else:
            growth_outlook = "Moderate or Low growth"
        return {
            "Company Name": company_name,
            "Current Price": info.get("regularMarketPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Growth Outlook": growth_outlook,
            "CEO Name": ceo_name
        }
    except Exception as e:
        return {"Error": f"Could not retrieve stock data: {e}"}

# ---------------------------
# Helper: Format change with one decimal (no emojis)
# ---------------------------
def format_change(val):
    if pd.isna(val):
        return ""
    return f"{val:+.1f}%"

# ---------------------------
# Compute Top Weekly Movers (from first 50 tickers) – using Weekly Change
# ---------------------------
def compute_top_weekly_movers(df: pd.DataFrame):
    # Use only the first 50 tickers from the CSV.
    df_top50 = df.head(50)

    def process_ticker(row):
        ticker_value = str(row["ticker"])
        sector = get_sector_for_ticker(ticker_value)
        hist = download_stock_history(ticker_value, period="3mo", interval="1d")
        if hist is None or len(hist) < 6:
            daily_change = None
            weekly_change = None
        else:
            close_series = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            daily_change = (
                ((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2]) * 100
                if len(close_series) >= 2 else None
            )
            weekly_change = (
                ((close_series.iloc[-1] - close_series.iloc[-6]) / close_series.iloc[-6]) * 100
                if len(close_series) >= 6 else None
            )
        return (
            ticker_value,
            row.get("company name", "N/A"),
            sector,
            format_change(daily_change) if daily_change is not None else "",
            format_change(weekly_change) if weekly_change is not None else ""
        )

    results = []
    total = len(df_top50)
    progress_bar = st.progress(0)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, row): idx for idx, row in df_top50.iterrows()}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            progress_bar.progress(completed / total)
    res_df = pd.DataFrame(results, columns=["Ticker", "Company Name", "Sector", "Daily Change", "Weekly Change"])
    res_df = res_df[res_df["Weekly Change"] != ""]
    res_df["WeeklyNum"] = (
        res_df["Weekly Change"]
        .str.replace("+", "")
        .str.replace("%", "")
        .astype(float)
        .abs()
    )
    res_df.sort_values(by="WeeklyNum", ascending=False, inplace=True)
    res_df.drop(columns=["WeeklyNum"], inplace=True)
    return res_df.head(10)

# ---------------------------
# Stock & CEO Analysis
# ---------------------------
def clean_ceo_name(ceo_name, company_name):
    ceo_name = ceo_name.replace("Mr.", "").replace("Ms.", "").replace("Mrs.", "").replace("Dr.", "").replace("Jr.", "").replace("Sr.", "").strip()
    name_parts = ceo_name.split()
    if len(name_parts) > 2:
        ceo_name = f"{name_parts[0]} {name_parts[-1]}"
    override_ceo = {
        "Amazon.com, Inc.": "Andy Jassy",
        "Tesla, Inc.": "Elon Musk",
        "Meta Platforms, Inc.": "Mark Zuckerberg"
    }
    return override_ceo.get(company_name, ceo_name)

def analyze_ceo(ceo_name):
    ceo_analysis = {
        "Elon Musk": "Known for visionary risk-taking, but also overconfidence leading to bold (often delayed) promises.",
        "Mark Zuckerberg": "Strong long-term vision; criticized for overconfidence in Metaverse pivot, causing volatility.",
        "Andy Jassy": "Pragmatic, focusing on AWS growth, but lacks Jeff Bezos's aggressive innovation reputation.",
        "Tim Cook": "Operational excellence and sustainability focus rather than high-risk innovation.",
        "Satya Nadella": "Strategic leadership revitalized Microsoft with cloud & AI focus; moderate risk-taker.",
        "Jeff Bezos": "Highly ambitious and competitive, known for relentless pursuit of innovation (now stepped down)."
    }
    return ceo_analysis.get(ceo_name, "CEO analysis not found in local database.")

def get_stock_info(ticker_symbol):
    return get_stock_info_cached(ticker_symbol)

def get_ceo_sentiment(company_name, ceo_name):
    if not NEWS_API_KEY:
        return "NEWS_API_KEY not set in environment. Cannot retrieve CEO sentiment."
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        query = f'"{ceo_name}" AND "{company_name}"'
        news_url = (
            f'https://newsapi.org/v2/everything?q={query}'
            f'&from={start_date}&to={end_date}'
            f'&sortBy=relevancy&apiKey={NEWS_API_KEY}'
        )
        response = requests.get(news_url).json()
        if response.get("status") != "ok":
            return f"News API Error: {response.get('message', 'Unknown error')}"
        articles = response.get("articles", [])[:10]
        if not articles:
            fallback_query = f'"{company_name} CEO"'
            fallback_url = (
                f'https://newsapi.org/v2/everything?q={fallback_query}'
                f'&from={start_date}&to={end_date}'
                f'&sortBy=relevancy&apiKey={NEWS_API_KEY}'
            )
            response = requests.get(fallback_url).json()
            articles = response.get("articles", [])[:10]
        if not articles:
            return "No recent CEO news found for sentiment analysis."
        sentiments = []
        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""
            text = title + ". " + description
            if text.strip():
                sentiment_score = sia.polarity_scores(text)["compound"]
                sentiments.append(sentiment_score)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        if avg_sentiment > 0.1:
            return "Positive sentiment towards CEO overall."
        elif avg_sentiment < -0.1:
            return "Negative sentiment towards CEO overall."
        else:
            return "Neutral sentiment towards CEO overall."
    except Exception as e:
        return f"Sentiment analysis failed: {e}"

def get_stock_analysis(ticker_symbol):
    stock_info = get_stock_info(ticker_symbol)
    if "Error" in stock_info:
        return f"Error: {stock_info['Error']}"
    ceo_name = stock_info.get("CEO Name", "N/A")
    company_name = stock_info.get("Company Name", "N/A")
    ceo_analysis = analyze_ceo(ceo_name)
    ceo_sentiment = get_ceo_sentiment(company_name, ceo_name)
    response = f"**Company Analysis for {company_name} ({ticker_symbol})**\n\n"
    for key, value in stock_info.items():
        response += f"- **{key}:** {value}\n"
    response += f"\n**CEO Cognitive Bias Analysis for {ceo_name}:**\n{ceo_analysis}\n"
    response += f"\n**CEO Sentiment Analysis:**\n{ceo_sentiment}"
    return response

# ---------------------------
# Market Overview with Daily and Monthly Trends
# ---------------------------
def get_daily_overview():
    market_tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT",
        "Dow 30": "^DJI"
    }
    overview = {}
    for index, ticker in market_tickers.items():
        data = get_index_history(ticker, period="2d")
        if data is None or len(data) < 2:
            overview[index] = {"Daily % Change": "N/A"}
            continue
        current_close = data["Close"][-1]
        prev_close = data["Close"][-2]
        pct_change = ((current_close - prev_close) / prev_close) * 100
        overview[index] = {"Daily % Change": round(pct_change, 2)}
    return overview

def get_monthly_overview():
    market_tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT",
        "Dow 30": "^DJI"
    }
    overview = {}
    for index, ticker in market_tickers.items():
        data = get_index_history(ticker, period="1mo")
        if data is None or len(data) < 2:
            overview[index] = {"Monthly % Change": "N/A"}
            continue
        first_close = data["Close"][0]
        last_close = data["Close"][-1]
        pct_change = ((last_close - first_close) / first_close) * 100
        overview[index] = {"Monthly % Change": round(pct_change, 2)}
    return overview

def interpret_overview(overview_data, key):
    pos = 0
    neg = 0
    for v in overview_data.values():
        pct = v.get(key)
        if isinstance(pct, (int, float)):
            if pct > 0:
                pos += 1
            elif pct < 0:
                neg += 1
    if pos > neg:
        return f"Overall, {key.lower()} indicates an upward trend."
    elif neg > pos:
        return f"Overall, {key.lower()} indicates a downward trend."
    else:
        return f"Overall, {key.lower()} is mixed."

def get_direction_emoji(pct):
    if isinstance(pct, (int, float)):
        if pct > 0:
            return "📈"
        elif pct < 0:
            return "📉"
        else:
            return "➖"
    return ""

def format_change_with_emoji(pct):
    if isinstance(pct, (int, float)):
        return f"{pct:+.2f}% {get_direction_emoji(pct)}"
    else:
        return pct

st.sidebar.title("🗺 Market Overview")
if st.sidebar.button("Check Market"):
    daily = get_daily_overview()
    monthly = get_monthly_overview()
    daily_interpretation = interpret_overview(daily, "Daily % Change")
    monthly_interpretation = interpret_overview(monthly, "Monthly % Change")

    df_daily = pd.DataFrame.from_dict(daily, orient="index").reset_index()
    df_daily.rename(columns={"index": "Index"}, inplace=True)
    df_monthly = pd.DataFrame.from_dict(monthly, orient="index").reset_index()
    df_monthly.rename(columns={"index": "Index"}, inplace=True)
    df_combined = pd.merge(df_daily, df_monthly, on="Index")

    df_combined["Daily % Change"] = df_combined["Daily % Change"].apply(format_change_with_emoji)
    df_combined["Monthly % Change"] = df_combined["Monthly % Change"].apply(format_change_with_emoji)

    st.sidebar.markdown("**Market Overview:**")
    st.sidebar.write(df_combined.to_html(index=False), unsafe_allow_html=True)
    st.sidebar.markdown(f"**Daily Trend:** {daily_interpretation}")
    st.sidebar.markdown(f"**Monthly Trend:** {monthly_interpretation}")
    st.sidebar.markdown("**Summary:** Overall, the daily and monthly trends combined suggest mixed signals.")

# ---------------------------
# Enrich Investment Query Function
# ---------------------------
def enrich_investment_query(user_text: str, df: pd.DataFrame) -> str:
    text_upper = user_text.upper()
    matched_row = None
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        company = str(row["company name"]).upper() if "company name" in row else ""
        if ticker in text_upper or company in text_upper:
            matched_row = row
            break
    if matched_row is not None:
        enriched = (
            f"{user_text}\n\nData from topbuzz500: "
            f"Ticker: {matched_row['ticker']}, "
            f"Company: {matched_row['company name']}, "
            f"Comments: {matched_row.get('comments', 'N/A')}. "
            "Please provide investment advice based on this data, including additional insights such as sector, market cap, and growth outlook."
        )
        return enriched
    else:
        return user_text

# ---------------------------
# Function to generate TradingView chart link for a stock
# ---------------------------
def get_chart_link(user_text: str, df: pd.DataFrame) -> str:
    text_upper = user_text.upper()
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        if ticker in text_upper:
            return f"https://www.tradingview.com/symbols/{ticker}/"
    return ""

# ---------------------------
# Main Chatbot UI (Investment Advice)
# ---------------------------
st.title("StockTalk: Your Investment Buddy")
if st.session_state["user"]:
    with st.form(key="chat_form", clear_on_submit=True):
        user_text = st.text_input("Your Investment Question:")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_text.strip():
        # SECURITY GUARDS
        if contains_disallowed_content(user_text):
            st.error("⚠️ Your message contains disallowed content or potential jailbreak attempts.")
        elif is_too_frequent_or_long(user_text):
            st.warning("🚫 Your message is too large or too frequent. Please wait or shorten your prompt.")
        else:
            # If content is safe and not too frequent, proceed
            enriched_query = enrich_investment_query(user_text, st.session_state["user_csv_db"])
            if conversation:
                response = conversation.run(enriched_query)
            else:
                response = "LLM not initialized. Provide an API key or select a model."

            chart_link = get_chart_link(user_text, st.session_state["user_csv_db"])
            if chart_link:
                response += f"\n\nFor visual insights, view the TradingView chart: [View on TradingView]({chart_link})"

            st.session_state["chat_history"].append({"user": user_text, "bot": response})
            st.write("🤖:", response)
else:
    st.warning("Please log in to start chatting.")

if st.session_state["chat_history"]:
    st.subheader("📝 Conversation History")
    for entry in st.session_state["chat_history"]:
        st.write(f"**You:** {entry['user']}")
        st.write(f"**Bot:** {entry['bot']}")
        st.write("---")

# ---------------------------
# Sidebar for Top Movers based on Weekly Moves
# ---------------------------
st.sidebar.title("🔥 High Volatility Stocks")
with st.sidebar.expander("ℹ Info"):
    st.write("Daily Change: % change between the last two trading days.")
    st.write("Weekly Change: % change between the last trading day and the day 5 days ago.")
    st.write("We show the top 10 stocks (from the first 50) with the biggest weekly moves.")
if st.sidebar.button("View Top Movers"):
    result_df = compute_top_weekly_movers(st.session_state["user_csv_db"])
    if result_df.empty:
        st.sidebar.warning("No valid ticker data or insufficient data for weekly move.")
    else:
        html_table = result_df.to_html(index=False)
        st.sidebar.write(html_table, unsafe_allow_html=True)

# ---------------------------
# Stock & CEO Analysis
# ---------------------------
st.sidebar.title("📊 Stock & CEO Analysis")
stock_symbol = st.sidebar.text_input("Enter stock ticker (e.g. TSLA, AMZN, AAPL):")
if st.sidebar.button("🔍 Check & Analyze"):
    if stock_symbol:
        analysis_result = get_stock_analysis(stock_symbol.upper())
        st.sidebar.markdown(analysis_result)
    else:
        st.sidebar.warning("Please enter a valid stock ticker.")

# ---------------------------
# Investment Calculator
# ---------------------------
st.subheader("📈 Investment Calculator")
principal = st.number_input("💵 Initial Investment ($)", min_value=0, value=1000)
rate = st.number_input("📊 Annual Return Rate (%)", min_value=0.0, max_value=50.0, value=7.0)
years = st.number_input("🗓️ Years", min_value=1, max_value=50, value=10)
if st.button("💰 Calculate Future Value"):
    fv = principal * ((1 + rate/100) ** years)
    st.write(f"💰 Future Value after {years} years: **${fv:,.2f}**")

# ---------------------------
# Quick News Checker with News Sentiment
# ---------------------------
def get_latest_news(query, max_articles=3):
    if not NEWS_API_KEY:
        return ["NEWS_API_KEY not set"]
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles
    }
    resp = requests.get(base_url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        articles = data.get("articles", [])
        summaries = []
        for article in articles:
            title = article.get("title", "No Title")
            url = article.get("url", "")
            sentiment_score = sia.polarity_scores(title)["compound"]
            if sentiment_score > 0.1:
                sentiment_indicator = "🔵"
            elif sentiment_score < -0.1:
                sentiment_indicator = "🔴"
            else:
                sentiment_indicator = "🟡"
            summaries.append(f"- {title} (Sentiment: {sentiment_score:.2f} {sentiment_indicator})\n  {url}")
        return summaries
    else:
        return [f"Error: {resp.status_code}"]

st.sidebar.title("📰 Quick News")
news_query = st.sidebar.text_input("Enter a topic/stock to get news:")
if st.sidebar.button("🔎 Get News"):
    news_results = get_latest_news(news_query)
    st.sidebar.write("**Top News Articles:**")
    for article in news_results:
        st.sidebar.write(article)

# ---------------------------
# Export Chat as JSON
# ---------------------------
if st.button("📥 Export Chat as JSON"):
    json_data = json.dumps(st.session_state["chat_history"], indent=4)
    st.download_button(label="Download JSON", data=json_data, file_name="chat_history.json", mime="application/json")

# ---------------------------
# Export Chat as PDF
# ---------------------------
def export_chat_to_pdf(chat_history):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    text_y = 750
    for entry in chat_history:
        user_text = f"You: {entry['user']}"
        bot_text = f"Bot: {entry['bot']}"
        c.drawString(50, text_y, user_text)
        text_y -= 20
        c.drawString(50, text_y, bot_text)
        text_y -= 40
        if text_y < 60:
            c.showPage()
            text_y = 750
    c.save()
    pdf_data = buf.getvalue()
    buf.close()
    return pdf_data

if st.button("📄 Export Chat as PDF"):
    pdf_data = export_chat_to_pdf(st.session_state["chat_history"])
    st.download_button(label="Download PDF", data=pdf_data, file_name="chat_history.pdf", mime="application/pdf")
