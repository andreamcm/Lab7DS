# ------------------------------------ #
# Universidad del Valle de Guatemala   #
# Autores:                             #
#   Andrea Cordón, 16076               #
#   Cristopher Recinos, 16005          #
#   Pablo Lopez, 14509                 #
# lab7.py                              #
# ------------------------------------ #


# Librerias
import numpy # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

import nltk
#nltk.download('wordnet') #Use this line once to download.
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

countA = 0
countB = 0
countC = 0
countD = 0
countE = 0
countF = 0

for t in pd_data['text']:
    for i in t:
        if i == "mixco":
            countA = countA + 1

for t in pd_data['text']:
    for i in t:
        if i == "roosevelt":
            countB = countB + 1

for t in pd_data['text']:
    for i in t:
        if i == "periferico":
            countC = countC + 1

for t in pd_data['text']:
    for i in t:
        if i == "cristóbal":
            countD = countD + 1

for t in pd_data['text']:
    for i in t:
        if i == "próceres":
            countE = countE + 1

for t in pd_data['text']:
    for i in t:
        if i == "lobos":
            countF = countF + 1
			
#print("La cantidad de accidentes reportados en la última semana es: ", countA)

print("Mayores zonas con trafico:")
print("Mixco: ", countA)
print("Roosevelt: ", countB)
print("Periferico: ", countC)
print("San Cristobal: ", countD)
print("proceres: ", countE)
print("Villa Lobos: ", countF)
