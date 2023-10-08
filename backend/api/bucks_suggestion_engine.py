from bucks_ai_engine import initiate
from bucks_cases_search import initiate as init_case_search
import pandas as pd

def syllabus_text(text):
    tfidf_df, word_freq_scores, entity_df, topic_dict =  initiate(text)
    top_topics = []
    # Top 2 topics from bag-of-words
    # Top 2 topics from topic modelling
    # 1 Entity from NER

    # Combining Top Cases
    top_suggested_cases = init_case_search(top_topics)
