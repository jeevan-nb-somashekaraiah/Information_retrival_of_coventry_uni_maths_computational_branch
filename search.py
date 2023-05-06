import json
import requests
import numpy as np
from time import sleep
from random import randint
from bs4 import BeautifulSoup
import re
import string
import pandas as pd

import math
from collections import defaultdict
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Define the inverted index
try:
    # Try to load the inverted index from a file
    with open('inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
except FileNotFoundError:
    # If the file doesn't exist, create a new inverted index
    inverted_index = defaultdict(set)

def crawler():
    url = 'https://pureportal.coventry.ac.uk/en/organisations/research-centre-for-computational-science-and-mathematical-modell/publications/?page='
    pages = np.arange(0, 5)
    publications = []
    for page in pages:
        page = str(page)
        _url = url + page
        web_page = requests.get(_url)
        
        soup = BeautifulSoup(web_page.content, "html.parser")
        documents = soup.findAll("div", {"class" : "result-container"})


        for doc in documents:
            title = doc.find("h3", {"class" : "title"})
            pub_link = title.find('a', {'class':'link'}).get('href')
            pub_page = requests.get(pub_link)
            pub_title = title.find('span').text
            date = doc.find('span',{"class": "date"}).text
            pub_links = requests.get(pub_link)
            soup1 = BeautifulSoup(pub_links.content,"html.parser")
            abst = soup1.find("div",{'class':"content-content publication-content"})
            abstract = abst.find("div",{'class':'textblock'})
            if abstract==None:
                abstract=''
            else:
                abstract=abstract.text
        
            try:
                authors = doc.findAll('a', {'class':'link person'})
                author_names = []
                author_links = []
                for author in authors:
                    author_names.append(author.find('span').text)
                    author_links.append(author.get('href'))
                
            
            except Exception as e:
                continue
            if not author_links:
                continue

            # Add the new document to the inverted index
            document_index = len(publications)
            publications.append([pub_title,pub_link,author_names,author_links,date,abstract])
            document_tokens = abstract.split()
            for token in document_tokens:
                inverted_index[token].add(document_index)
            
            # Save the inverted index to a file
            with open('inverted_index.pkl', 'wb') as f:
                pickle.dump(inverted_index, f)
    
    return publications
publications = crawler()
#Save the crawled data to a JSON file
with open('publications.json', 'w') as f:
  json.dump(publications, f)


nltk.download('stopwords')
# Define the stop words to be removed
stop_words = set(stopwords.words('english'))

# Define the stemmer to be used
stemmer = SnowballStemmer('english')
# Define the list to store the cleaned and stemmed documents
documents_clean_stemmed = []

documents_clean = []
for d in publications:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', str(d))
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Lowercase the numbers
    document_test = re.sub(r'[0-9]', '', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    documents_clean.append(document_test)
     # Remove stop words
    document_test = ' '.join([word for word in document_test.split() if word not in stop_words])
    # Apply stemming
    document_test_stemmed = ' '.join([stemmer.stem(word) for word in document_test.split()])
    documents_clean_stemmed.append(document_test_stemmed)
    documents_clean_stemmed.append(document_test_stemmed)

# Define the inverted index
    inverted_index = defaultdict(set)
# create an inverted index

for i, document in enumerate(documents_clean_stemmed):
    for word in document.split():
        inverted_index[word].append(i)
with open('inverted_index.pkl', 'wb') as f:
            pickle.dump(inverted_index, f)


def search(query):
  
  #preprocessing for query
  query_tokens = query.split()
  query_tokens = [item.lower() for item in query_tokens]
  new_words = []
  for word in query_tokens:
      for letter in word:
          if letter in string.punctuation:
              word = word.replace(letter,"")   
      new_words.append(word)
  query_tokens = new_words
  query_tokens = [s.replace('  ', ' ') for s in query_tokens]
  query_tokens= [s.encode('ascii', 'ignore').decode() for s in query_tokens]
  # calculate the tf-idf scores for the query

  tf_idf_scores = defaultdict(float)
  for token in query_tokens:
      if token in inverted_index:
          idf = math.log(len(documents_clean) / len(inverted_index[token]))
          for document_index in inverted_index[token]:
              tf = documents_clean[document_index].split().count(token)
              tf_idf_scores[document_index] += tf * idf

  # get the top matching documents
  
  results = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
  # return the top matching documents
  return results, documents_clean_stemmed,publications

