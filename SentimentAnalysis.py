#!/usr/bin/env python3
"""
ANLP Project - Customer Support Sentiment Analysis
Data Exploration and Analysis Script
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')
 
# QUICK FIX for Google Colab NLTK issue - Run this first!
def fix_nltk_colab():
    """Quick fix for NLTK tokenizer issue in Colab"""
    import nltk
    print("🔧 Fixing NLTK tokenizer issue...")
   
    # Download all required NLTK data
    downloads = ['punkt_tab', 'punkt', 'stopwords', 'averaged_perceptron_tagger']
   
    for item in downloads:
        try:
        	nltk.download(item, quiet=True)
        	print(f"✅ Downloaded {item}")
        except:
        	print(f"⚠️  Could not download {item}")
   
    print("✅ NLTK fix complete! Now run the main analysis.")
 
# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)  # Updated tokenizer
    nltk.download('punkt', quiet=True)      # Fallback for older versions
    nltk.download('stopwords', quiet=True)
except:
    pass
 
class CustomerSupportAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
       
    def load_amazon_data(self, filepath):
        """Load and preprocess Amazon reviews dataset"""
        print("Loading Amazon Reviews dataset...")
        df = pd.read_csv(filepath)
       
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
       
        return df
   
    def load_twitter_data(self, filepath):
        """Load and preprocess Twitter customer support dataset"""
        print("Loading Twitter Customer Support dataset...")
        df = pd.read_csv(filepath)
       
        # Fix boolean parsing for inbound column
        df['inbound'] = df['inbound'].map({'True': True, 'False': False, True: True, False: False})
       
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample structure:")
        print(df.head(2))
       
        return df
   
    def analyze_amazon_ratings(self, df):
        """Analyze rating distributions in Amazon data"""
        print("\n=== AMAZON REVIEWS ANALYSIS ===")
       
        # Rating distribution
        rating_counts = df['Score'].value_counts().sort_index()
        print("\nRating Distribution:")
        for rating, count in rating_counts.items():
        	percentage = (count / len(df)) * 100
        	print(f"{rating} Stars: {count:,} reviews ({percentage:.1f}%)")
       
        # Sentiment grouping
        def categorize_sentiment(score):
        	if score >= 4:
            	return 'Positive'
        	elif score == 3:
            	return 'Neutral'
        	else:
            	return 'Negative'
       
        df['Sentiment'] = df['Score'].apply(categorize_sentiment)
        sentiment_counts = df['Sentiment'].value_counts()
       
        print("\nSentiment Grouping:")
        for sentiment, count in sentiment_counts.items():
        	percentage = (count / len(df)) * 100
        	print(f"{sentiment}: {count:,} reviews ({percentage:.1f}%)")
       
        # Plot rating distribution
        plt.figure(figsize=(12, 5))
       
        plt.subplot(1, 2, 1)
        rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Amazon Reviews - Rating Distribution')
        plt.xlabel('Star Rating')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=0)
       
        plt.subplot(1, 2, 2)
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgray', 'lightgreen'])
        plt.title('Sentiment Distribution')
        plt.ylabel('')
       
        plt.tight_layout()
        plt.show()
       
        return df
   
    def analyze_text_structure(self, df, text_column, dataset_name):
        """Analyze text structure and statistics"""
        print(f"\n=== {dataset_name.upper()} TEXT STRUCTURE ANALYSIS ===")
       
        # Remove null values
        df_clean = df[df[text_column].notna()].copy()
       
        # Text length analysis
        df_clean['text_length'] = df_clean[text_column].str.len()
        df_clean['word_count'] = df_clean[text_column].str.split().str.len()
       
        print(f"\nText Length Statistics:")
        print(f"Average characters: {df_clean['text_length'].mean():.1f}")
        print(f"Median characters: {df_clean['text_length'].median():.1f}")
        print(f"Average words: {df_clean['word_count'].mean():.1f}")
        print(f"Median words: {df_clean['word_count'].median():.1f}")
        print(f"Shortest text: {df_clean['text_length'].min()} characters")
        print(f"Longest text: {df_clean['text_length'].max()} characters")
       
        # Plot text length distributions
        plt.figure(figsize=(12, 4))
       
        plt.subplot(1, 2, 1)
        plt.hist(df_clean['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{dataset_name} - Character Length Distribution')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.axvline(df_clean['text_length'].mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
       
        plt.subplot(1, 2, 2)
        plt.hist(df_clean['word_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title(f'{dataset_name} - Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.axvline(df_clean['word_count'].mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
       
        plt.tight_layout()
        plt.show()
       
        return df_clean
   
    def clean_text_for_analysis(self, text):
        """Clean text for word frequency analysis"""
        if pd.isna(text):
        	return ""
       
        # Convert to lowercase
        text = text.lower()
       
        # Remove URLs, mentions, hashtags for social media text
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
       
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
       
        # Tokenize - try NLTK first, fallback to simple split
        try:
        	tokens = word_tokenize(text)
        except:
        	# Simple fallback tokenization
        	tokens = text.split()
       
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
       
        return ' '.join(tokens)
   
    def word_frequency_analysis(self, df, text_column, dataset_name, sentiment_column=None):
        """Analyze word frequencies and create word clouds"""
        print(f"\n=== {dataset_name.upper()} WORD FREQUENCY ANALYSIS ===")
       
        # Clean all text
        df['cleaned_text'] = df[text_column].apply(self.clean_text_for_analysis)
       
        # Overall word frequency
        all_text = ' '.join(df['cleaned_text'].dropna())
        word_freq = Counter(all_text.split())
       
        print(f"\nTop 10 Most Common Words:")
        for word, freq in word_freq.most_common(10):
        	print(f"{word}: {freq:,} occurrences")
       
        # Create overall word cloud
        plt.figure(figsize=(15, 5))
       
        plt.subplot(1, 3, 1)
        if len(all_text) > 0:
        	wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
        	plt.imshow(wordcloud, interpolation='bilinear')
        	plt.title(f'{dataset_name} - Overall Word Cloud')
        	plt.axis('off')
       
        # Sentiment-specific analysis if available
        if sentiment_column and sentiment_column in df.columns:
        	sentiments = df[sentiment_column].unique()
        	
        	for i, sentiment in enumerate(['Positive', 'Negative']):
            	if sentiment in sentiments:
                	sentiment_text = ' '.join(df[df[sentiment_column] == sentiment]['cleaned_text'].dropna())
                	
                	if len(sentiment_text) > 0:
                    	plt.subplot(1, 3, i+2)
                    	wordcloud = WordCloud(width=400, height=300, background_color='white').generate(sentiment_text)
                    	plt.imshow(wordcloud, interpolation='bilinear')
                    	plt.title(f'{sentiment} Sentiment Words')
                    	plt.axis('off')
                    	
                    	# Print top words for this sentiment
                    	sentiment_freq = Counter(sentiment_text.split())
                    	print(f"\nTop 5 {sentiment} Words:")
                    	for word, freq in sentiment_freq.most_common(5):
                        	print(f"  {word}: {freq:,}")
       
        plt.tight_layout()
        plt.show()
       
        return word_freq
   
    def twitter_specific_analysis(self, df):
        """Twitter-specific analysis for customer support data"""
        print("\n=== TWITTER CUSTOMER SUPPORT SPECIFIC ANALYSIS ===")
       
        # Twitter features analysis first
        df['has_mention'] = df['text'].str.contains('@', na=False)
        df['has_hashtag'] = df['text'].str.contains('#', na=False)
        df['has_url'] = df['text'].str.contains('http', na=False)
       
        # Message type analysis (after adding feature columns)
        customer_msgs = df[df['inbound'] == True]
        company_msgs = df[df['inbound'] == False]
       
        print(f"Customer messages (inbound=True): {len(customer_msgs):,} ({len(customer_msgs)/len(df)*100:.1f}%)")
        print(f"Company responses (inbound=False): {len(company_msgs):,} ({len(company_msgs)/len(df)*100:.1f}%)")
       
        # Company analysis
        if len(company_msgs) > 0:
        	companies = company_msgs['author_id'].unique()
        	print(f"\nNumber of companies: {len(companies)}")
        	print("Companies:", companies[:10])  # Show first 10
        	
        	# Company response frequency
        	company_counts = company_msgs['author_id'].value_counts()
        	print(f"\nTop 5 Most Active Companies:")
        	for company, count in company_counts.head().items():
            	print(f"  {company}: {count:,} responses")
       
        print(f"\nTwitter Features Usage:")
        print(f"Messages with @mentions: {df['has_mention'].sum():,} ({df['has_mention'].mean()*100:.1f}%)")
        print(f"Messages with #hashtags: {df['has_hashtag'].sum():,} ({df['has_hashtag'].mean()*100:.1f}%)")
        print(f"Messages with URLs: {df['has_url'].sum():,} ({df['has_url'].mean()*100:.1f}%)")
       
        # Compare customer vs company usage
        print(f"\nCustomer vs Company Feature Usage:")
        print(f"Customer @mentions: {customer_msgs['has_mention'].mean()*100:.1f}%")
        print(f"Company @mentions: {company_msgs['has_mention'].mean()*100:.1f}%")
        print(f"Customer URLs: {customer_msgs['has_url'].mean()*100:.1f}%")
        print(f"Company URLs: {company_msgs['has_url'].mean()*100:.1f}%")
       
        # Visualizations
        plt.figure(figsize=(15, 5))
       
        # Message type distribution
        plt.subplot(1, 3, 1)
        message_types = ['Customer\nMessages', 'Company\nResponses']
        message_counts = [len(customer_msgs), len(company_msgs)]
        plt.pie(message_counts, labels=message_types, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        plt.title('Message Type Distribution')
       
        # Twitter features
        plt.subplot(1, 3, 2)
        features = ['@Mentions', '#Hashtags', 'URLs']
        percentages = [df['has_mention'].mean() * 100, df['has_hashtag'].mean() * 100, df['has_url'].mean() * 100]
        bars = plt.bar(features, percentages, color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Twitter Features Usage')
        plt.ylabel('Percentage of Messages')
        plt.ylim(0, 100)
        for bar, v in zip(bars, percentages):
        	plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{v:.1f}%', ha='center')
       
        # Company activity (top 10)
        plt.subplot(1, 3, 3)
        if len(company_msgs) > 0:
        	top_companies = company_counts.head(8)
        	plt.barh(range(len(top_companies)), top_companies.values, color='orange', alpha=0.7)
            plt.yticks(range(len(top_companies)), [comp[:15] + '...' if len(comp) > 15 else comp for comp in top_companies.index])
        	plt.xlabel('Number of Responses')
        	plt.title('Most Active Companies')
        	plt.gca().invert_yaxis()
       
        plt.tight_layout()
        plt.show()
       
        return df
 
def main():
    """Main execution function for Google Colab"""
    analyzer = CustomerSupportAnalyzer()
   
    print("Customer Support Sentiment Analysis - Data Exploration")
    print("=" * 60)
   
    # Load Twitter dataset from Colab files
    print("Loading Twitter Customer Support dataset...")
    try:
        twitter_df = analyzer.load_twitter_data('twcs.csv')
       
        # Perform Twitter analysis
        print("\n" + "="*60)
        twitter_analyzed = analyzer.analyze_text_structure(twitter_df, 'text', 'Twitter Support')
       
        print("\n" + "="*60)
        analyzer.word_frequency_analysis(twitter_df, 'text', 'Twitter Support')
       
        print("\n" + "="*60)
        twitter_final = analyzer.twitter_specific_analysis(twitter_analyzed)
       
        print(f"\n✅ Twitter Dataset Successfully Loaded: {len(twitter_df):,} messages")
       
    except FileNotFoundError:
        print("⚠️  Twitter dataset file not found. Make sure 'twcs.csv' is uploaded to Colab.")
   
    # Load Amazon dataset from Colab files
    print("\n" + "="*60)
    print("Loading Amazon Reviews dataset...")
    try:
        amazon_df = analyzer.load_amazon_data('Reviews.csv')
       
        # Perform Amazon analysis
        print("\n" + "="*60)
        amazon_analyzed = analyzer.analyze_amazon_ratings(amazon_df)
       
        print("\n" + "="*60)
        amazon_text_analyzed = analyzer.analyze_text_structure(amazon_analyzed, 'Text', 'Amazon Reviews')
       
        print("\n" + "="*60)
        analyzer.word_frequency_analysis(amazon_analyzed, 'Text', 'Amazon Reviews', 'Sentiment')
       
        print(f"\n✅ Amazon Dataset Successfully Loaded: {len(amazon_df):,} reviews")
       
    except FileNotFoundError:
        print("⚠️  Amazon dataset file not found. Make sure 'Reviews.csv' is uploaded to Colab.")
    except Exception as e:
        print(f"⚠️  Error loading Amazon dataset: {e}")
        print("	Note: Large datasets may take time to load in Colab")
   
    # Combined analysis and insights
    if 'twitter_df' in locals() and 'amazon_df' in locals():
        print("\n" + "="*60)
        print("=== COMBINED DATASET INSIGHTS FOR PRESENTATION ===")
       
        customer_msgs = twitter_df[twitter_df['inbound'] == True]
        company_msgs = twitter_df[twitter_df['inbound'] == False]
       
        print(f"📊 TWITTER DATASET:")
        print(f"   • Total messages: {len(twitter_df):,}")
        print(f"   • Customer messages: {len(customer_msgs):,}")
        print(f"   • Company responses: {len(company_msgs):,}")
        print(f"   • Companies represented: {len(company_msgs['author_id'].unique())}")
       
        print(f"\n📊 AMAZON DATASET:")
        print(f"   • Total reviews: {len(amazon_df):,}")
        print(f"   • Rating distribution: {dict(amazon_df['Score'].value_counts().sort_index())}")
       
        sentiment_counts = amazon_analyzed['Sentiment'].value_counts()
        print(f"   • Sentiment distribution: {dict(sentiment_counts)}")
       
        print(f"\n🎯 PROJECT ADVANTAGES:")
        print(f"   ✅ Discriminative Task: {len(amazon_df):,} labeled reviews for sentiment classification")
        print(f"   ✅ Generative Task: {min(len(customer_msgs), len(company_msgs)):,} conversation pairs for response generation")
        print(f"   ✅ Cross-domain Evaluation: Test sentiment models across review→support domains")
        print(f"   ✅ Real-world Scale: Large datasets for robust training")
       
        # Data quality insights
        print(f"\n📈 DATA QUALITY INSIGHTS:")
        twitter_avg_len = twitter_df['text'].str.len().mean()
        amazon_avg_len = amazon_df['Text'].str.len().mean()
        print(f"   • Twitter avg message length: {twitter_avg_len:.0f} characters")
        print(f"   • Amazon avg review length: {amazon_avg_len:.0f} characters")
        print(f"   • Text diversity: Excellent for robust NLP model training")
   
    print("\n" + "="*60)
    print("✅ DATA EXPLORATION COMPLETE!")
    print("✅ Ready for Week 6 presentation with comprehensive dataset analysis")
    print("✅ Next steps: Implement preprocessing pipeline and baseline models")
 
# Additional Colab-specific functions
def setup_colab_environment():
    """Setup function for Google Colab environment"""
    import subprocess
    import sys
   
    print("Setting up Google Colab environment...")
   
    # Install required packages
    packages = ['wordcloud', 'nltk']
    for package in packages:
        try:
        	__import__(package)
        	print(f"✅ {package} already installed")
        except ImportError:
        	print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
   
    # Download NLTK data
    import nltk
    try:
        nltk.download('punkt_tab', quiet=True)  # Updated tokenizer for newer NLTK
        nltk.download('punkt', quiet=True)      # Fallback for older versions
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except:
        print("⚠️  NLTK download failed - word tokenization may not work")
   
    print("🚀 Environment setup complete!")
 
def check_file_sizes():
    """Check the size of uploaded datasets"""
    import os
   
    files_to_check = ['twcs.csv', 'Reviews.csv']
   
    print("📁 Dataset File Information:")
    for filename in files_to_check:
        if os.path.exists(filename):
        	size_mb = os.path.getsize(filename) / (1024 * 1024)
        	print(f"   {filename}: {size_mb:.1f} MB")
        else:
        	print(f"   {filename}: ❌ Not found")
 
def quick_preview():
    """Quick preview of both datasets"""
    import pandas as pd
   
    print("🔍 Quick Dataset Preview:")
   
    # Twitter preview
    try:
        twitter_sample = pd.read_csv('twcs.csv', nrows=5)
        print(f"\n📱 Twitter Dataset (twcs.csv):")
        print(f"   Columns: {list(twitter_sample.columns)}")
        print(f"   Sample text: '{twitter_sample['text'].iloc[0][:100]}...'")
    except:
        print("\n📱 Twitter dataset not accessible")
   
    # Amazon preview 
    try:
        amazon_sample = pd.read_csv('Reviews.csv', nrows=5)
        print(f"\n🛒 Amazon Dataset (Reviews.csv):")
        print(f"   Columns: {list(amazon_sample.columns)}")
        if 'Text' in amazon_sample.columns:
        	print(f"   Sample review: '{amazon_sample['Text'].iloc[0][:100]}...'")
    except:
        print("\n🛒 Amazon dataset not accessible")
 
if __name__ == "__main__":
    main()
 
# Additional utility functions for specific analyses
 
def create_presentation_plots():
    """Create additional plots specifically for presentation"""
   
    # Model comparison template
    plt.figure(figsize=(10, 6))
    models = ['Logistic\nRegression', 'Naive\nBayes', 'SVM', 'LSTM', 'BERT']
    accuracies = [0.78, 0.76, 0.82, 0.87, 0.91]  # Example values
   
    bars = plt.bar(models, accuracies, color=['lightblue', 'lightgreen', 'lightcoral', 'orange', 'purple'])
    plt.title('Expected Model Performance Comparison')
    plt.ylabel('Expected Accuracy')
    plt.ylim(0, 1)
   
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            	f'{acc:.1%}', ha='center', va='bottom')
   
    plt.tight_layout()
    plt.show()
 
def project_timeline_visualization():
    """Create project timeline visualization"""
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
   
    # Project timeline
    tasks = ['Data Exploration', 'Preprocessing', 'Baseline Models', 'Neural Models', 'Evaluation', 'Report']
    start_dates = [datetime(2025, 6, 10) + timedelta(days=i*7) for i in range(6)]
    durations = [7, 10, 14, 14, 7, 7]  # days
   
    plt.figure(figsize=(12, 6))
   
    for i, (task, start, duration) in enumerate(zip(tasks, start_dates, durations)):
        plt.barh(i, duration, left=start, height=0.6,
                color=plt.cm.viridis(i/len(tasks)), alpha=0.8)
        plt.text(start + timedelta(days=duration/2), i, task,
            	ha='center', va='center', fontweight='bold')
   
    plt.xlabel('Timeline')
    plt.ylabel('Project Tasks')
    plt.title('ANLP Project Timeline')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)
    plt.yticks([])
    plt.tight_layout()
    plt.show()
 
# Google Colab Execution
if __name__ == "__main__":
    print("🚀 Starting Customer Support Sentiment Analysis Project")
    print("="*70)
   
    # Step 0: Fix NLTK first
    fix_nltk_colab()
   
    # Step 1: Setup environment
    setup_colab_environment()
   
    # Step 2: Check uploaded files
    print("\n" + "="*70)
    check_file_sizes()
   
    # Step 3: Quick preview
    print("\n" + "="*70)
    quick_preview()
   
    # Step 4: Full analysis
    print("\n" + "="*70)
    main()
   
    # Step 5: Additional visualizations for presentation
    print("\n" + "="*70)
    create_presentation_plots()
    project_timeline_visualization()
   
    print("\n🎉 ALL ANALYSIS COMPLETE!")
    print("📋 Use the generated plots and statistics for your Week 6 presentation")
    print("📂 Your findings are ready to copy into your presentation slides")

