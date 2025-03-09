# 🤖 StockTalk: Your AI Investment Buddy

## Introduction

StockTalk is an interactive investment analysis app built using Streamlit and LangChain, allowing users to interact with AI-driven financial insights. The goal was to integrate various financial tools into an interactive platform, providing users with real-time stock insights, market trends, sentiment analysis, and AI-powered investment discussions, making financial information more accessible and engaging.

## Features

- 📈 Market Overview – Get insights into major indices like S&P 500, Nasdaq, and Dow Jones.

- 💬 AI Investment Chatbot – Ask questions about stocks, market trends, and investment strategies.

- 📊 Stock & CEO Analysis – Fetch company and CEO insights, including sentiment analysis.

- 🔥 High Volatility Stocks – View the most volatile stocks over the past week.

- 📥 Export Chat History – Save your chat conversation as JSON or PDF.

- 🔎 Real-time News Analysis – Get the latest stock-related news and sentiment insights.

- 📄 Investment Calculator – Calculate future investment returns based on compound interest.


## Data Overview

### API Usage

The app relies on **OpenAI’s GPT API** to generate and evaluate interview questions dynamically.
Ensure you have a valid **OpenAI API key** to use this tool.

---

## Technologies Used

- Streamlit – Interactive web app framework.
- Yahoo Finance API (yfinance) – Retrieves stock market data.
- NewsAPI – Fetches the latest news for sentiment analysis.
- Firebase – User authentication and session management.
- OpenAI API (langchain) – AI-powered investment chatbot.
- Matplotlib – Stock data visualization.
- NLTK & TextBlob – Sentiment analysis on news articles.


## Installation and Setup

1. Clone the Repository

```markdown
git clone git@github.com:TuringCollegeSubmissions/aumilas-AE.2.5.git
```

2. Create a Virtual Environment (Optional)

```markdown
pip install -r requirements.txt
```python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```markdown
pip install -r requirements.txt
```

4. Set Up API Keys

Create a .env file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_key
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key  # (Optional)
```

Also, ensure you have your Firebase credentials (firebase_credentials.json) in the root directory.

5. Run the Application

```markdown
streatlit run Trading_app.py
```

## Authors

**Aura Milasiute** - [GitHub](https://github.com/auramila)

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgments

Thank you [Turing College](https://www.turingcollege.com)
