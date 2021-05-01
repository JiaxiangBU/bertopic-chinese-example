# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + id="Fa7NsaukRLro"
import os

# +
# # !pip install openpyxl
# -

import pandas as pd
df = pd.read_excel("data/chinese-sts.xlsx",skiprows=1,engine='openpyxl',names=['pair_l','pair_r','scores'])

# xl
df = df[df['pair_l'].str.len()>=2][pd.isna(df['pair_l'])==False]

# +
# df.head()
# -

input_list = df['pair_l']

# + colab={"base_uri": "https://localhost:8080/"} id="W0r_mgEvT9nh" outputId="548e12ea-0c58-4a19-8102-93ae439194fd"
# # !pip install sentence_transformers
# -

# %ls *distilroberta*

# + id="hlxNa-jdRhvo"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilroberta-fine-tuned/') # use 'paraphrase-distilroberta-base-v1'

# + colab={"base_uri": "https://localhost:8080/"} id="WWBxMKOFUNiS" outputId="02b391ab-5d8c-4c2d-be13-e8b15d55fd14"
# # !pip install bertopic

# +
# # !pip install -U scikit-learn

# +
# https://stackoverflow.com/questions/65816675/cannot-import-name-delayed-from-sklearn-utils-fixes
# # !pip install delayed

# + id="uCzhbj9MULhG"
from bertopic import BERTopic

# + id="KGao5whzT3ky"
topic_model = BERTopic(embedding_model=model)

# + id="jAW4HAL5UVwm"
docs = input_list.tolist()

# + colab={"base_uri": "https://localhost:8080/"} id="hH3oykGAU7qB" outputId="15a99d6f-a6ca-48a5-f393-ebad69ba2149"
len(docs)
# -

embeddings = model.encode(docs)

embedding_df = pd.DataFrame(embeddings)

embedding_df.head()

# + id="W-0olgxKUIGs"
topic_model = BERTopic()
# ipdb.set_trace()
topics, _ = topic_model.fit_transform(docs[:1000],embeddings=embedding_df.iloc[:1000,].values)
# 20k数据，就很慢了。
# -

import bertopic, scipy, numpy

for i in [bertopic, scipy, numpy]:
    print(i.__name__,i.__version__)

import numpy as np

np.any(embedding_df.iloc[:100,].isin([np.nan, np.inf, -np.inf])), \
np.any(pd.Series(docs[:100]).isin([np.nan, np.inf, -np.inf,""," "])), \

embedding_df.shape,len(docs)


