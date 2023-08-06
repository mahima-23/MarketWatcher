import json
import urllib.parse
from newspaper import Article, ArticleException
import en_core_web_md
from django.shortcuts import render
from django.http import HttpRequest
from transformers import pipeline, AutoTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import numpy as np
from django.http import HttpResponse
from .models import Company

import os


# Load the spaCy model
nlp = en_core_web_md.load()

def fetch_news(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def scrape_news(request: HttpRequest):
    # API key for GNews API
    gnews_api_key = str(os.getenv('GNEWS_API_KEY'))


    # List of companies
    company_list = [
        "ICICI Bank",
        "Reliance Industries",
        "HDFC Bank",
        "Infosys",
        "TCS",
        "JSW Steel",
        "SBI Bank",
        "Bharti Airtel",
        "Axis Bank",
        "Tata Motors",

    ]

    results = []

    for company in company_list:
        # Fetch news articles from GNews API
        gnews_encoded_company = urllib.parse.quote(company)
        gnews_url = f"https://gnews.io/api/v4/top-headlines?q={gnews_encoded_company}&lang=en&country=in&max=10&apikey={gnews_api_key}"
        gnews_data = fetch_news(gnews_url)
        try:
                gnews_articles = gnews_data["articles"]
        except KeyError:
                print(f"KeyError occurred for company: {company}")
                continue

        if gnews_articles:
            for gnews_article in gnews_articles:
                gnews_headline = gnews_article["title"]
                gnews_article_url = gnews_article["url"]
                gnews_published = gnews_article["publishedAt"]
                gnews_image = gnews_article["image"]

                # Use newspaper3k to extract article content for GNews API
                article = Article(gnews_article_url)
                try:
                    article.download()
                    article.parse()
                    content = article.text
                except ArticleException as e:
                    print(f"Failed to download article: {e}")
                    content = None

                result = {
                    "Company": company,
                    "Source": "GNews",
                    "Title": gnews_headline,
                    "Link": gnews_article_url,
                    "Content": content if content else None,
                    "Published_At": gnews_published,
                    "Image": gnews_image,
                }

                results.append(result)

                # Adding a delay of 1 second between requests
                time.sleep(1)

    # Load tokenizer for summarization
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    summary_list = []

    for result in results:
        try:
            content = result['Content']
            if content:
                # Truncate the content if it exceeds the maximum sequence length
                if len(content) > 1024:
                    content = content[:1024]

                # Tokenize the content
                tokenized_text = tokenizer.encode(content, truncation=True, max_length=1024, return_tensors="pt")

                # Perform summarization using transformers pipeline
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

                # Generate summaries in chunks of maximum sequence length
                chunked_text = [tokenizer.decode(chunk) for chunk in tokenized_text]
                summaries = []

                for chunk in chunked_text:
                    summaries.append(summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'])

                # Concatenate the summaries
                summary = " ".join(summaries)
                result['summary'] = summary
            else:
                result['summary'] = "No summary available"
        except IndexError:
            result['summary'] = "No summary available"

        if result['summary']:
            # Load the sentiment analyzer
            sentiment_analyzer = SentimentIntensityAnalyzer()
            sentiment_score = sentiment_analyzer.polarity_scores(result['summary'])
            compound_score = sentiment_score['compound']

            if compound_score >= 0.05:
                result['sentiment'] = 'Positive'
            elif compound_score <= -0.05:
                result['sentiment'] = 'Negative'
            else:
                result['sentiment'] = 'Neutral'
        else:
            result['sentiment'] = 'No sentiment available'

    # Pass the data to the template for rendering
    context = {'news_list': results, 'company_list': company_list}
    return render(request, 'scrape_news.html', context)




from urllib.parse import quote_from_bytes

def search_company(request):
    # Query the Company model to retrieve results for the specified company
    company = request.GET['company_name']
    gnews_encoded_company = urllib.parse.quote(company)

     # API key for GNews API
    gnews_api_key = str(os.getenv('GNEWS_API_KEY'))

   

    results = []

    # Fetch news articles from GNews API
    # gnews_encoded_company = urllib.parse.quote(results)
    gnews_url = f"https://gnews.io/api/v4/top-headlines?q={gnews_encoded_company}&lang=en&country=in&max=10&apikey={gnews_api_key}"
    gnews_data = fetch_news(gnews_url)
    try:
        gnews_articles = gnews_data["articles"]
    except KeyError:
        print(f"KeyError occurred for company: {company}")

    if gnews_articles:
        for gnews_article in gnews_articles:
            gnews_headline = gnews_article["title"]
            gnews_article_url = gnews_article["url"]
            gnews_published = gnews_article["publishedAt"]
            gnews_image = gnews_article["image"]

            # Use newspaper3k to extract article content for GNews API
            article = Article(gnews_article_url)
            try:
                article.download()
                article.parse()
                content = article.text
            except ArticleException as e:
                print(f"Failed to download article: {e}")
                content = None

            result = {
                    "Company": company,
                    "Source": "GNews",
                    "Title": gnews_headline,
                    "Link": gnews_article_url,
                    "Content": content if content else None,
                    "Published_At": gnews_published,
                    "Image": gnews_image,
            }

            results.append(result)

           
     # Load tokenizer for summarization
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    summary_list = []

    for result in results:
        try:
            content = result['Content']
            if content:
                # Truncate the content if it exceeds the maximum sequence length
                if len(content) > 1024:
                    content = content[:1024]

                # Tokenize the content
                tokenized_text = tokenizer.encode(content, truncation=True, max_length=1024, return_tensors="pt")

                # Perform summarization using transformers pipeline
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

                # Generate summaries in chunks of maximum sequence length
                chunked_text = [tokenizer.decode(chunk) for chunk in tokenized_text]
                summaries = []

                for chunk in chunked_text:
                    summaries.append(summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'])

                # Concatenate the summaries
                summary = " ".join(summaries)
                result['summary'] = summary
            else:
                result['summary'] = "No summary available"
        except IndexError:
            result['summary'] = "No summary available"

        if result['summary']:
            # Load the sentiment analyzer
            sentiment_analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = sentiment_analyzer.polarity_scores(result['summary'])
            compound_score = sentiment_scores['compound']

            if compound_score >= 0.05:
                result['sentiment'] = 'Positive'
            elif compound_score <= -0.05:
                result['sentiment'] = 'Negative'
            else:
                result['sentiment'] = 'Neutral'
        else:
            result['sentiment'] = 'No sentiment available'


    # Prepare the data to be displayed and pass it to the appropriate template
    context = {
       'news_list': results,
       'company_name':company
    }

    # Render the HTML template with the data and return as the response
    return render(request, 'search_results.html', context)







