# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 18:50:49 2021

@author: Vishwa
"""
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.util import mark_negation
from string import punctuation
from IPython.display import display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#%%
def penn_to_wn(tags):
    
    if tags.startswith('J'):
        return wn.ADJ
    elif tags.startswith('N'):
        return wn.NOUN
    elif tags.startswith('R'):
        return wn.ADV
    elif tags.startswith('V'):
        return wn.VERB
    return None

#%%
def get_sentiment_score(text):
    total_score=0
    raw_sentences=sent_tokenize(text)
    for sentence in raw_sentences:
        sent_score=0
        sentence=sentence.replace("<br />"," ").translate(str.maketrans('','',punctuation)).lower()
        tokens=TreebankWordTokenizer().tokenize(text)
        tags=pos_tag(tokens)
        for word,tag in tags:
            wn_tag=penn_to_wn(tag)
            if not wn_tag:
                continue
            lemma=WordNetLemmatizer().lemmatize(word,pos=wn_tag)
            if not lemma:
                continue
            synsets=wn.synsets(lemma,pos=wn_tag)
            if not synsets:
                continue
            synset=synsets[0]
            swn_synset=swn.senti_synset(synset.name())
            sent_score += swn_synset.pos_score() - swn_synset.neg_score()
        total_score = total_score + (sent_score / len(tokens))
    return (total_score / len(raw_sentences)) * 100

#%%

reviews=pd.read_csv(r"D:/Projects/NLP/1. Sentiment Analysis and Natural Language Processing for Marketing/undersamplereview.csv")

#%%
print(reviews.shape)

#%%
reviews.dropna(subset=['reviewText'], inplace=True)
#%%
nltk.download('punkt')

#%%
reviews['swn_score'] = reviews['reviewText'].apply(lambda text : get_sentiment_score(text))

#%%

fig , ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
sns.histplot(x='swn_score', data=reviews.query("swn_score < 8 and swn_score > -8"), ax=ax)
plt.show()

#%%
reviews['swn_sentiment'] = reviews['swn_score'].apply(lambda x: "positive" if x>1 else ("negative" if x<0.5 else "neutral"))
reviews['swn_sentiment'].value_counts(dropna=False)
sns.countplot(x='overall', hue='swn_sentiment' ,data = reviews)
sns.boxenplot(x='swn_sentiment', y='overall', data = reviews)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,7))
sns.boxenplot(x='overall', y='swn_score', data = reviews, ax=ax)
plt.show()

#%%
reviews['true_sentiment'] = \
    reviews['overall'].apply(lambda x: "positive" if x>=4 else ("neutral" if x==3 else "negative"))
y_swn_pred, y_true = reviews['swn_sentiment'].tolist(), reviews['true_sentiment'].tolist()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_swn_pred)
fig , ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
sns.heatmap(cm, cmap='viridis_r', annot=True, fmt='d', square=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True');





















