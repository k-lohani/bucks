import requests
import pandas as pd
# !pip install scholarly
from scholarly import scholarly
import json

def return_json(dictt):
  json_string = json.dumps(dictt)
  return json_string

def get_related_articles(title, num = 10):
    top_articles = {'title': [], 'abstract': []}
    search_query = scholarly.search_pubs(title)
    while len(top_articles['title']) < num:
      try:
          publication_info = next(search_query)
          top_articles['title'].append(publication_info['bib']['title'])
          top_articles['abstract'].append(publication_info['bib']['abstract'])
      except StopIteration:
          # top_articles['title'].append(None)
          # top_articles['abstract'].append(None)
          return top_articles
    return top_articles


# # TESTING
# title = 'Banking and Climate Change'
# res = get_related_articles(title, 20)
# display(pd.DataFrame(res))

def initiate(topics):
  all_top_articles_by_topics = {'title': [], 'abstract': []}
  for topic in topics:
    top_related_articles = get_related_articles(topic, num =1)
    all_top_articles_by_topics['title'].append("".join(top_related_articles['title']))
    all_top_articles_by_topics['abstract'].append("".join(top_related_articles['abstract']))


  return all_top_articles_by_topics

# TESTING
# topics = ["banking and climate change", 'Logistics Management and Climate Change']
# res = initiate(topics)
# print(pd.DataFrame(res['top_related_articles']))

# Unused Code Below

def get_more_sources_query(query):
  api_url = f"https://api.crossref.org/works?query={query}"

  response = requests.get(api_url)
  data = response.json()

  crossref_dict = {'doi':[], 'title':[]}
  for i in data["message"]["items"]:
    crossref_dict['doi'].append(i['DOI'])
    if 'title' in i:
      crossref_dict['title'].append(" ".join(i['title']))
    else:
      crossref_dict['title'].append("NA")

  return crossref_dict

def get_doi_info(doi):
    # Define the Unpaywall API endpoint
    api_url = f"https://api.unpaywall.org/v2/{doi}?email=kaustubhlohani25@gmail.com"

    try:
        # Make the API request to Unpaywall
        response = requests.get(api_url)
        data = response.json()

        # Check if the DOI is found and has content
        if "title" in data:
            title = data["title"]
        else:
            title = "Title not found"

        if "abstract" in data:
            abstract = data["abstract"]
        else:
            abstract = "Abstract not found"

        return title, abstract

        return data

    except requests.exceptions.RequestException as e:
        # Handle API request errors
        print(f"Error fetching data for DOI {doi}: {e}")
        return None, None

