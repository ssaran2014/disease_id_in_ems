import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import json

import os
import pathlib

def upload_dataset():
    df = pd.read_csv(os.path.join('.', 'data', 'combined_datasource.csv'), header=0)
    return df

def remove_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation)) 



class Disease():
    """Takes in symptoms and outputs associated disease"""

    #upload dataset of issues, symptoms and urls
    df = upload_dataset()

    #create a bag of words and tokenize
    bag_of_words = [word for word in df.symptoms]
    tf=TfidfVectorizer(lowercase=True, stop_words='english', analyzer = 'char', ngram_range=(2,4))
    text_tf= tf.fit_transform(bag_of_words)

    def __init__(self, symptoms):
        self.symptoms = symptoms

    def top_disease(self):

        #matrix multiplication to identify top diseases
        w = self.tf.transform([self.symptoms])
        response = np.matmul(self.text_tf.toarray(), np.transpose(w.toarray()))
        response_index = np.argmax(response)
        
        top_issue = self.df.iloc[response_index]['disease']
        top_issue_symptoms = self.df.iloc[response_index]['symptoms']
        top_issue_url = self.df.iloc[response_index]['url']

        #listing out the other top five alternatives
        response_series = pd.Series(response.squeeze())
        top_five_series = response_series.sort_values(ascending=False)[1:5]
        top_five_index = top_five_series.index
        top_five = '; '.join(list(self.df.disease.iloc[top_five_index]))
        return dict({
                        "speech_recognition": str(self.symptoms),
                        "top_issue": str(top_issue),
                        "top_issue_symptom": str(top_issue_symptoms),
                        "top_issue_url": str(top_issue_url),
                        "top_five": str(top_five)})
