# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:46:22 2022

@author: mrvis
"""

import ndjson
import pandas as pd
import numpy as np
import seaborn as sns

#%%
with open(r'C:\Users\mrvis\Documents\Sentiment-Analysis-NLP-for-Marketting-main\creating_dataset\Video_Games_5.json\Video_Games_5.json') as f:
    data = ndjson.load(f)

#%%
r_df=pd.DataFrame(data)
print(r_df.head())

# =============================================================================
# Data Dictionry
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# vote - helpful votes of the review
# style - a disctionary of the product metadata, e.g., "Format" is "Hardcover"
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# image - images that users post after they have received the product
# =============================================================================
#%%
print(r_df.shape)

#%%
print(r_df.info())

#%%
sns.countplot(data=r_df, x='overall')

#%%
print(len(r_df['asin'].value_counts(dropna=False)))

#%%
#taking 1500 for rating 1, 500 for rating 2,3,4 and 1500 for 5 rating

one_1500=r_df[r_df['overall']==1].sample(n=1500)
two_500=r_df[r_df['overall']==2].sample(n=500)
three_500=r_df[r_df['overall']==3].sample(n=500)
four_500=r_df[r_df['overall']==4].sample(n=500)
five_1500=r_df[r_df['overall']==5].sample(n=1500)

#%%

undersamplereview=pd.concat([one_1500,two_500,three_500,four_500,five_500],axis=0)
#%%
print(undersamplereview['overall'].value_counts(dropna=False))
print(undersamplereview.info())

#%%
sns.countplot(data=undersamplereview,x='overall')

#%%
#random Sapling 1,00,000
sample_100k_revs=r_df.sample(n=100000,random_state=42)
sample_100k_revs.to_csv('sample_100k_revs.csv', index=False)
undersamplereview.to_csv('undersamplereview.csv', index=False)





















