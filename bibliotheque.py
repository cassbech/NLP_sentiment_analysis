#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import pandas as pd
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from unicodedata import *
from unidecode import *
import unicodedata
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


# In[18]:


def remove_url(tweet):
    """
    retirer les url.
    """
    liste_element=tweet.split(" ")
    liste_clean=[]
    for elemt in liste_element:
        if not (elemt.startswith("http")):
            liste_clean.append(elemt)
    return " ".join(liste_clean)
    
    

def remplace_accents(tweet):
    """
    remplacer les accents.
    """
    try:
        s1 = unidecode(tweet)
        s2 = unicodedata.normalize('NFD', s1).encode('ascii', 'ignore')
        return str(s2,'utf-8')
    except:
        return("error")
        
def remove_ponctuation(tweet):
    """
    Supprimer les ponctuations.
    """
    list_punct=list(string.punctuation)
    tweets_without_punc=[]
    for char in tweet:
        if char not in list_punct:
            tweets_without_punc.append(char)
        else:
            tweets_without_punc.append(" ")
    return "".join(tweets_without_punc)  
    
def remove_at(tweet):
    """
    Supprimer les identification de users "@...".
    """
    liste_element=tweet.split(" ")
    liste_clean=[]
    for elemt in liste_element:
        if not (elemt.startswith("@")):
            liste_clean.append(elemt)
    return " ".join(liste_clean)
    
def filter_tab(tweet):
    """
    Supprimer les tabulations.
    """
    liste_clean = []
    liste_mots = tweet.split(" ")
    for mot in liste_mots:
        if not ("\n" in mot):
            liste_clean.append(mot)
        else:
            liste_clean.append(mot.replace("\n"," "))
    return " ".join(liste_clean)

def filter_tab(tweet):
    """
    Supprimer les tabulations.
    """
    liste_clean = []
    liste_mots = tweet.split(" ")
    for mot in liste_mots:
        if not ("\n" in mot):
            liste_clean.append(mot)
        else:
            liste_clean.append(mot.replace("\n"," "))
    return " ".join(liste_clean)

def remove_stop_words(tweet):
    stop_words = set(stopwords.words('french'))  
    word_tokens = word_tokenize(tweet)  
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
 
    return(" ".join(filtered_sentence)) 

def word_occurency(data_frame):
    """
    Liste de l'ensemble des mots utilisés dans les tweets
    """
    liste_de_mots = []
    for tweet in data_frame["Tweets traités"]:
        for mot in tweet.split(" "):
            liste_de_mots.append(mot)
    words_occurency = pd.DataFrame(pd.Series(liste_de_mots).value_counts()[1:],columns=["occurence"])
    distincts_word = words_occurency.index.tolist()
    return(distincts_word)

def matrix_of_termes(data_frame,distincts_word):
    corps = [tweets for tweets in data_frame["Tweets traités"]]
    df = pd.DataFrame(data=corps, columns=['Tweets'])
    vectorizer = CountVectorizer(vocabulary=distincts_word, min_df=0,stop_words=frozenset(), token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(df['Tweets'].values)
    result = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    
    result["Index"]=result.index
    data_frame["Index"]=data_frame.index
    result["Index"]=result["Index"].astype(int)
    data_frame["Index"] = data_frame["Index"].astype(int)
    
    data_avec_matrice_de_termes = data_frame.merge(result, left_on='Index', right_on='Index')
    data_avec_matrice_de_termes.drop(columns="Index",inplace=True)
    return data_avec_matrice_de_termes

def create_data_frame(data):
    nom = input("Quel nom voulez vous donner à votre DataFrame ? : ")
    nom+= ".csv"
    data.to_csv(nom)
    print("Bravo ton tableau est prêt à l'emploi tu es un AS, Bon machine learning ! 😀... Pour information le fichier enregistré est au format csv")


# In[ ]:




