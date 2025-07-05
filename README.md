# NLP-SentimentAnalysis
A data exploration and sentiment analysis project using Amazon Reviews and Twitter Customer Support datasets. Includes visualizations, word clouds, and Twitter-specific insights. Built for ANLP coursework with full Colab compatibility and support for NLTK preprocessing.
# 🧠 Customer Support Sentiment Analysis – ANLP Project

This repository contains a comprehensive data exploration and sentiment analysis pipeline for two datasets:

1. 🛒 **Amazon Reviews** (`Reviews.csv`)
2. 📱 **Twitter Customer Support** (`twcs.csv`)

Developed as part of an **Advanced Natural Language Processing (ANLP)** university project, the goal is to analyze customer sentiments, text structures, and support interactions across both platforms.

---

## 📊 Features

- 📦 **Amazon Ratings Analysis**
  - Distribution of star ratings
  - Grouping into Positive / Neutral / Negative sentiments
  - Sentiment-specific word cloud visualizations

- 🐦 **Twitter Support Analysis**
  - Message type breakdown: Customer vs Company
  - Text feature usage: mentions, hashtags, and URLs
  - Top responding companies and their activity
  - Text length, structure, and keyword trends

- 📚 **Text Cleaning and Tokenization**
  - Lowercasing, punctuation/URL removal
  - Stopword filtering using NLTK
  - Word frequency extraction using `collections.Counter`

- 📈 **Visualizations**
  - Histograms for length distributions
  - Pie charts for sentiment and message types
  - Bar charts for company activity and model performance
  - Word clouds for overall and sentiment-specific vocab

- 🧪 **Colab Compatibility**
  - NLTK fixes for Google Colab
  - Auto-download missing libraries and data
  - File checks and previews included

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `anlp_sentiment_analysis.py` | Main analysis script with class-based design |
| `twcs.csv` | Twitter Customer Support dataset |
| `Reviews.csv` | Amazon Product Reviews dataset |
| `README.md` | Project overview and instructions |

---

## 🚀 Getting Started

### 📌 Prerequisites

- Python 3.6+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk
  - wordcloud

Install with:
```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud
