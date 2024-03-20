from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

def get_most_similar_name_and_corresponding_result(text, references):
    vectorizer = TfidfVectorizer()
    max = 0
    most_similar = ''
    for store_name in references:
        vectors = vectorizer.fit_transform([text, store_name])
        similarity = cosine_similarity(vectors)
        similarity_score = similarity[0][1]
        if similarity_score > max:
            max = similarity_score
            most_similar = store_name
    try:
        numerical_rst_string = text.split(':')[1].split(';')[0]
    except:
        numerical_rst_string = "0"
        print(f"Error: cannot extract from text ###{text}###! max: {max}, most_similar: {most_similar}.")
    return most_similar, numerical_rst_string, max

def extract_from_string(n):
    try:
        return int(n)
    except:
        if n == 'N/A':
            return 0
        try:
            n = n.split('%')[0]
            return int(n)
        except:
            print(f"Error: cannot extract from n ###{n}###!")
            n = 0
            return n
        
def get_vote(x, groups):
    rst_down = 0
    rst_up = 0
    rst_same = 0
    for group in groups:
        if x['store_temp'] > float(groups[group]['High']):
            rst_down += x[group] # too hot
        elif x['store_temp'] < float(groups[group]['Low']):
            rst_up += x[group] # too cold
        else:
            rst_same += x[group] # just right
    return np.array([rst_down, rst_up, rst_same])