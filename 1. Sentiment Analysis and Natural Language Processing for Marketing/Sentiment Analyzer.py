# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:20:57 2022

@author: mrvis
"""

# =============================================================================
# Creating a Dictionary-based Sentiment Analyzer
# =============================================================================

import pandas as pd
import nltk
from IPython.display import display
pd.set_option('display.max_columns', None)

#%%
#importing 100k data set created using create dataset file

reviews=pd.read_csv('sample_100k_revs.csv')
print(reviews.head())

#%%

# =============================================================================
# Tokenziation, we will use different tokenizer and compare them
# =============================================================================

from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
import string
#%%
tb = TreebankWordTokenizer()

reviews["rev_text_lower"] = reviews['reviewText'].apply(lambda rev: str(rev)\
                                                        .translate(str.maketrans('', '', punctuation))\
                                                        .replace("<br />", " ")\
                                                        .lower())
    
#%%
print(reviews[['reviewText','rev_text_lower']].sample(2))
#%%

reviews["tb_tokens"]=reviews['rev_text_lower'].apply(lambda rev:tb.tokenize(str(rev)))

#%%
print(reviews[['tb_tokens','rev_text_lower']].sample(4))

#%%
pd.set_option('display.max_colwidth', None)

#%%

# =============================================================================
# Casual Tokenizer--- Just a different tokenizer
# =============================================================================

from nltk.tokenize.casual import casual_tokenize
#cs=casual_tokenize()

reviews['casualToken']=reviews['rev_text_lower'].apply(lambda rev:casual_tokenize(str(rev)))

#%%

print(reviews[['casualToken','tb_tokens']].sample(2))

#%%

import spacy

#%%
spacynlp = spacy.load("en_core_web_sm")

#%%
reviews['spacytoke']=reviews['rev_text_lower'].apply(lambda rev:spacynlp.tokenizer(rev))

#%%
print(reviews['spacytoke'].sample(2))

#%%

# =============================================================================
# Stemming process
# =============================================================================

from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

reviews['token_stemmed']=reviews['tb_tokens'].apply(lambda words:[stemmer.stem(w) for w in words])

#%%

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

#%%
# =============================================================================
# function to Convert between the PennTreebank tags to simple Wordnet tags
# =============================================================================

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

#%%

lemmatizer = WordNetLemmatizer()
def get_lemas(tokens):
    lemmas = []
    for token in tokens:
        pos = penn_to_wn(pos_tag([token])[0][1])
        if pos:
            lemma = lemmatizer.lemmatize(token, pos)
            if lemma:
                lemmas.append(lemma)
    return lemmas
#%%
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
#%%

#reviews['lemmas']=reviews['tb_tokens'].apply(lambda tokens:get_lemas(tokens))
reviews['lemmas'] = reviews['tb_tokens'].apply(lambda tokens: get_lemas(tokens)) 


#%%

# =============================================================================
# Sentiment Predictor Baseline Model
# =============================================================================

def get_sentiment_score(tokens):
    score = 0
    tags = pos_tag(tokens) #getting tags from tokens like walk,verb
    for word, tag in tags:
        wn_tag = penn_to_wn(tag) #converting pos_tags to wn like wn.ADJ
        if not wn_tag:
            continue
        synsets = wn.synsets(word, pos=wn_tag) #Getting Synonym for word
        if not synsets:
            continue
        
        #most common set:
        synset = synsets[0] #getting first synonym
        swn_synset = swn.senti_synset(synset.name()) #getting neg_score and pos_score
        
        score += (swn_synset.pos_score() - swn_synset.neg_score()) #calculting score 
        
    return score
       
#%%
nltk.download('sentiwordnet')

#%%
##test

print(swn.senti_synset(wn.synsets("perfect",wn.ADJ)[0].name()).pos_score())

#%%
reviews['sentiment_score']=reviews['lemmas'].apply(lambda tokens: get_sentiment_score(tokens))#calculting score

#%%
print(reviews[['reviewText','lemmas','sentiment_score']].sample(5))














