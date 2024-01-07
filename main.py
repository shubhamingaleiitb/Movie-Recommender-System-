#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise  import cosine_similarity


# # Data Collection and Pre-Processing

# In[30]:


data = pd.read_csv("movies.csv")
data.shape


# In[31]:


data.head()


# In[32]:


features = ["genres", "keywords", "tagline", "cast", "director"]

for feature in features:
    data[feature] = data[feature].fillna('')

combined_features = data['genres']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']


# In[33]:


combined_features


# In[34]:


vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)


# # Cosine Similarity

# In[35]:


similarity = cosine_similarity(feature_vectors)
similarity


# In[36]:


movie_name = input('Enter your favourite movie: ')


# In[37]:


list_of_all_movies = data['title'].to_list()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_movies)
print(find_close_match)


# In[38]:


close_match = find_close_match[0]
print(close_match)


# In[39]:


index_of_movie = data[data.title == close_match]['index'].values[0]
index_of_movie


# In[40]:


similarity_score = list(enumerate(similarity[index_of_movie]))
len(similarity_score)


# In[41]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[42]:


print("Movies suggested for you: \n")

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = data[data.index == index]['title'].values[0]
    if (i<30):
        print(i, '.',title_from_index)
        i+=1


# In[ ]:


import streamlit as st

movie = st.text_area("Enter a movie")

