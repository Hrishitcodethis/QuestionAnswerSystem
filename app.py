import streamlit as st
import json
import re
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the JSON file and preprocess articles (executes once)
file_path = 'news.article.json'
with open(file_path, 'r', encoding='utf-8') as file:
    articles = json.load(file)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Preprocess articles
cleaned_articles = []
for article in articles:
    cleaned_text = clean_text(article['articleBody'])
    cleaned_articles.append({
        'title': article['title'],
        'content': cleaned_text,
        'source': article['source']
    })

df = pd.DataFrame(cleaned_articles)

# Function to check if an article is relevant
def is_relevant(text):
    keywords = ['israel', 'hamas', 'gaza', 'palestine', 'war', 'conflict']
    return any(keyword in text for keyword in keywords)

# Filter relevant articles (executes once)
df['is_relevant'] = df['content'].apply(is_relevant)
relevant_articles = df[df['is_relevant']]

# Load tokenized corpus for BM25 (executes once)
with open('tokenized_corpus.pkl', 'rb') as f:
    tokenized_corpus = pickle.load(f)

# Load BM25 instance (executes once)
with open('bm25_instance.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Load the saved T5 model and tokenizer (executes once)
model_path = "saved_model/"
t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to retrieve relevant articles
def retrieve_articles(query, bm25, articles, top_n=5):
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = scores.argsort()[-top_n:][::-1]
    return articles.iloc[top_n_indices]

# Function to generate answers using T5 model
def question_answer_system(question, top_articles):
    context = ' '.join(top_articles['content'].tolist())
    inputs = t5_tokenizer.encode("question: " + question + " context: " + context, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=True)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app
def main():
    st.title("Question Answering System")

    user_question = st.text_input("Please enter your question:")
    if user_question:
        # Filter articles by relevance (optional, based on user preference)
        # Example: filtered_articles = relevant_articles[relevant_articles['source'] == 'example_source']
        
        # Apply search filters (optional, based on user preference)
        # Example: filtered_articles = filtered_articles[filtered_articles['date'] > '2023-01-01']
        
        top_articles = retrieve_articles(user_question, bm25, relevant_articles)
        answer = question_answer_system(user_question, top_articles)
        
        # Display answer
        st.subheader("Answer:")
        st.write(answer)
        
        # Display relevant links
        st.subheader("Relevant Links:")
        for article_title, article_source in zip(top_articles['title'], top_articles['source']):
            st.write(f"- [{article_title}]({article_source})")

if __name__ == "__main__":
    main()
