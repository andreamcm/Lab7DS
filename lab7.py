# ------------------------------------ #
# Universidad del Valle de Guatemala   #
# Autores:                             #
#   Andrea Cordón, 16076               #
#   Cristopher Recinos, 16005          #
# lab7.py                              #
# ------------------------------------ #


# Librerias
import numpy # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import FreqDist

# from wordcloud import WordCloud, STOPWORDS

# import matplotlib.pyplot as plt
# from matplotlib.ticker import StrMethodFormatter

# from textblob import TextBlob


# Lectura de los datos
pd_data = pd.read_csv('data.csv')
pd_data = pd_data[['text']]

print(pd_data.head(3))

# Puntuacion
def remove_punctuation(text): 
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# Stopwords
def remove_stopwords(text): 
    words = [w for w in text if w not in stopwords.words('english')]
    return words

# Lemmatizer
lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text): 
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

#Stemmer
stemmer = PorterStemmer()

def word_stemmer(text): 
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text


# Quitar puntuacion
tokenizer = RegexpTokenizer(r'\w+')

pd_data['text'] = pd_data['text'].apply(lambda x: tokenizer.tokenize(x.lower()))

# Quitar stopwords
pd_data['text'] = pd_data['text'].apply(lambda x: remove_stopwords(x))

# Lemmatizer
pd_data['text'].apply(lambda x: word_lemmatizer(x))

# Stemmer
pd_data['text'].apply(lambda x: word_stemmer(x))

print(pd_data.head(3))