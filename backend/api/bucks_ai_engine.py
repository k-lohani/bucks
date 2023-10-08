import nltk
nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import NMF
import json
import Levenshtein

# import custom json data files
def load_json(file):
  with open(file, 'r') as f:
    data = json.load(f)
  return data

# Cleaning the text

def jd_cleaning(text):
  # lowercasing
  text = text.lower()
  # HTML Tag removal
  soup = BeautifulSoup(text, 'html.parser')
  text = soup.get_text()
  # Special Character Removal
  text = re.sub(r'[^\w\s]', '', text)
  # Whitespace and Newline Removal
  text = text.replace('\n', ' ').replace('\t', '').replace('  ', ' ').strip()

  return str(text)

def jd_preprocess(text):
  # Tokenization
  words = word_tokenize(text)
  # Stopword Removal
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if w not in stop_words]
  # Lemmatization
  nlp = spacy.load('en_core_web_sm')
  words = " ".join(words)
  lem_obj = nlp(words)
  lemmatized_words = [token.lemma_ for token in lem_obj]
  # Noise Removal
  # spell check

  return lemmatized_words, lem_obj, words

def keyword_tf_idf(words):
  words = [" ".join(words)]
  tfidf = TfidfVectorizer()
  tfidf_matrix = tfidf.fit_transform(words)
  feature_names = tfidf.get_feature_names_out()

  tfidf_df = pd.DataFrame({
      'feature_names' : feature_names,
      'scores' : tfidf_matrix.toarray()[0]
  })
  tfidf_df.sort_values(by = 'scores', ascending = False, inplace = True)
  tfidf_df['scores'] = tfidf_df['scores'].round(3)
  # display(tfidf_df)

  # # plotting tf_idf_features and scores
  # plt.figure(figsize=(8, 6))
  # plt.barh(y = tfidf_df.feature_names[0:30], width = tfidf_df.scores[0:30])
  # plt.title('TF-IDF Scores')
  # plt.xlabel('Scores')
  # plt.ylabel('Keywords')

  return tfidf_df, tfidf_matrix, feature_names

# Word frequency analysis
def word_freq_analysis(words):
  word_freq = Counter(words)
  common_words = word_freq.most_common(10)
  word, freq = zip(*common_words)
  common_words_dict = {'keyword':[], 'frequency':[]}
  for w in common_words:
    common_words_dict['keyword'].append(w[0])
    common_words_dict['frequency'].append(w[1])
  common_words_df = pd.DataFrame(common_words_dict)
  return common_words_df

# Named Entity Recognition
def ner(words):
  text = " ".join(words)
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(text)
  entities = {'Entity Name': [], 'Entity Label': []}
  for ent in doc.ents:
    entities["Entity Name"].append(ent.text)
    entities['Entity Label'].append(ent.label_)
  entity_df = pd.DataFrame(entities)
  return entity_df

# Topic Modelling using NMF
def topic_modeling(tfidf_matrix, feature_names):
  nmf_model = NMF(n_components=5, random_state=1)
  nmf_topics = nmf_model.fit_transform(tfidf_matrix)
  topic_dict = {'topic_idx': [], 'top_words': []}
  for topic_idx, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topic_dict['topic_idx'].append(topic_idx)
    topic_dict['top_words'].append(top_words)
  topic_df = pd.DataFrame(topic_dict)
  print(topic_df)
  return topic_df

# skills and qualification extraction

# Function to check similarity
def lev_similar(word1, word2, threshold):
    distance = Levenshtein.distance(word1, word2)
    max_length = max(len(word1), len(word2))
    similarity = (max_length - distance) / max_length
    return similarity

def initiate(text):
  # cleaning text
  clean_text = jd_cleaning(text)
  # preprocessing text
  preprocessed_text, lem_obj, words = jd_preprocess(clean_text)
  # tfidf
  tfidf_df, tfidf_matrix, feature_names = keyword_tf_idf(preprocessed_text)
  # word frequency analysis
  word_freq_scores = word_freq_analysis(preprocessed_text)
  # named entity recognition
  entity_df = ner(preprocessed_text)
  # Topic Modeling
  topic_dict = topic_modeling(tfidf_matrix, feature_names)

  return tfidf_df, word_freq_scores, entity_df, topic_dict




# ************************************************************Testing**********************************************************************************************************************