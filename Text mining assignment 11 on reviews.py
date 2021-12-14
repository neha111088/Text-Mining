#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# ### WEB SCRAPING AND SENTIMENT ANALYSIS

# #### Extract reviews of any product from ecommerce website like amazon 
# 
# #### Perform emotion mining

# #### IMPORTING LIBRARIES

# In[1]:



from bs4 import BeautifulSoup as bs #Beautiful Soup is a Python library for pulling data out of HTML and XML files.
import requests # making HTTP requests in Python


# READING DATA

# In[2]:


bt='https://www.amazon.in'
ul='https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_getr_d_paging_btm_next_30?ie=UTF8&reviewerType=all_reviews'


# In[3]:


cust_name = []   #define list to store Name of the customers
review_title = []
rate = []
review_content = []


# In[4]:


tt = 0
while tt == 0:
    page = requests.get(ul)
    while page.ok == False:#if it fails to connect then this loop will be executing continuously until get response from site  
        page = requests.get(ul)
   

    soup = bs(page.content,'html.parser')
    soup.prettify()       #Prettify() function in BeautifulSoup will enable us to view how the tags are nested in the document.
    
    names = soup.find_all('span', class_='a-profile-name')
    names.pop(0)
    names.pop(0)
    
    for i in range(0,len(names)):
        cust_name.append(names[i].get_text())
        
    title = soup.find_all("a",{"data-hook":"review-title"})
    for i in range(0,len(title)):
        review_title.append(title[i].get_text())

    rating = soup.find_all('i',class_='review-rating')
    rating.pop(0)
    rating.pop(0)
    for i in range(0,len(rating)):
        rate.append(rating[i].get_text())

    review = soup.find_all("span",{"data-hook":"review-body"})
    for i in range(0,len(review)):
        review_content.append(review[i].get_text())
        
    try:
        for div in soup.findAll('li', attrs={'class':'a-last'}):
            A = div.find('a')['href']
        ul = bt + A
    except:
        break


# In[5]:


len(cust_name)


# In[6]:


len(review_title)


# In[7]:


len(review_content)


# In[8]:


len(rate)


# In[9]:


review_title[:] = [titles.lstrip('\n') for titles in review_title]

review_title[:] = [titles.rstrip('\n') for titles in review_title]

review_content[:] = [titles.lstrip('\n') for titles in review_content]

review_content[:] = [titles.rstrip('\n') for titles in review_content]


# In[10]:


get_ipython().system('pip install -U textblob')
get_ipython().system('python -m textblob.download_corpora')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob, Word, Blobber
import wordcloud
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
nltk.download('stopwords')


# In[11]:


df = pd.DataFrame()


# In[12]:


df['Customer Name'] = cust_name
df['Review Title'] = review_title
df['Rating'] = rate
df['Reviews'] = review_content


# In[13]:


df.head(10)


# In[14]:


df.to_csv(r'E:fill.csv',index = True)


# In[15]:


data = pd.read_csv("E:fill.csv",index_col=[0])


# In[16]:


data.dtypes


# In[17]:


data['Rating'] = [titles.rstrip(' out of 5 stars') for titles in data['Rating']]


# In[18]:


data['Rating']


# In[19]:


data['Rating'].value_counts(normalize=True)*100


# In[20]:


ratings=data.groupby(['Rating']).count()
ratings


# In[21]:


plt.figure(figsize=(12,8))
data['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[22]:


data.Rating.hist()
data.Rating.hist(bins=10)
plt.xlabel('Rating')
plt.ylabel('Count')


# In[23]:


data.iloc[:,[3]]


# In[24]:



Reviews=data.iloc[:,[3]]


# In[25]:


Reviews.shape


# In[26]:


Reviews.describe()


# In[27]:


Reviews.dtypes


# In[28]:


# removing customer name and reviw title column as they have not that significance in output##
data.drop(["Customer Name","Review Title"],axis=1,inplace=True)

data.head()


# In[32]:


data.Reviews.isna().sum()


# In[33]:


data['Reviews']=data['Reviews'].fillna(" ")


# In[34]:


data.Reviews.isna().sum()


# In[35]:


## Cleaning the text input for betting understanding of Machine..##

##Converting all review into Lowercase..###

data['Reviews']= data['Reviews'].apply(lambda x: " ".join(word.lower() for word in x.split()))


# In[36]:


## removing punctuation from review..#
import string
data['Reviews']=data['Reviews'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))


# In[37]:


## Remove Numbers from review...##
data['Reviews']=data['Reviews'].str.replace('[0-9]','')


# In[38]:


## removing all stopwords(english)....###
from nltk.corpus import stopwords


# In[39]:


stop_words=stopwords.words('english')


# In[40]:


data['Reviews']=data['Reviews'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))


# In[41]:


data.head(5)


# In[42]:


from textblob import Word
data['Reviews']= data['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

import re
pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
data['Reviews']= data['Reviews'].apply(lambda x:(re.sub(pattern, '',x).strip()))


# In[43]:


data['Reviews'].head()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer


vec = CountVectorizer()
X = vec.fit_transform(data['Reviews'])
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
TFIDF=tfidf.fit_transform(data['Reviews'])
print(TFIDF)


# In[46]:



Review_wordcloud = ' '.join(data['Reviews'])
Q_wordcloud=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(Review_wordcloud)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud)


# # Removing Punctuation
# The next step is to remove punctuation, as it doesn’t add any extra information while treating text data. Therefore removing all instances of it will help us reduce the size of the training data.

# In[47]:


data['Reviews'] = data['Reviews'].str.replace('[^\w\s]','')
data['Reviews'].head()


# #  Common word removal
# Previously, we just removed commonly occurring words in a general sense. We can also remove commonly occurring words from our text data First, let’s check the 10 most frequently occurring words in our text data then take call to remove or retain.

# In[48]:


freq = pd.Series(' '.join(data['Reviews']).split()).value_counts()[:10]
freq


# # Now, let’s remove these words as their presence will not of any use in classification of our text data.

# In[49]:



data['Reviews'] = data['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Reviews'].head()


# #  Rare words removal
# Similarly, just as we removed the most common words, this time let’s remove rarely occurring words from the text. Because they’re so rare, the association between them and other words is dominated by noise. You can replace rare words with a more general form and then this will have higher counts

# In[50]:


freq = pd.Series(' '.join(data['Reviews']).split()).value_counts()[-10:]
freq


# In[51]:


from textblob import TextBlob
data['Reviews'][:10].apply(lambda x: str(TextBlob(x).correct()))


# # 2.7 Tokenization
# Tokenization refers to dividing the text into a sequence of words or sentences. In our example, we have used the textblob library to first transform our reviews into a blob and then converted them into a series of words.

# In[52]:


TextBlob(data['Reviews'][0]).words


# In[53]:


TextBlob(data['Reviews'][1]).words


# # Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. For this purpose, we will use PorterStemmer from the NLTK library.

# In[54]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Reviews'][:10].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# # Lemmatization
# Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices. It makes use of the vocabulary and does a morphological analysis to obtain the root word. Therefore, we usually prefer using lemmatization over stemming.

# In[55]:


from textblob import Word
data['Reviews'] = data['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Reviews'].head()


# # Advance Text Processing
# Up to this point, we have done all the basic pre-processing steps in order to clean our data. Now, we can finally move on to extracting features using NLP techniques.
# 
#  
# # N-grams
# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.
# 
# Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure, like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to particular cases.

# In[56]:


TextBlob(data['Reviews'][0]).ngrams(2)


# # Term frequency
# Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.
# 
# Therefore, we can generalize term frequency as:
# 
# TF = (Number of times term T appears in the particular row) / (number of terms in that row)

# In[57]:


tf1 = (data['Reviews'][1:10]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# # Inverse Document Frequency
# The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.
# 
# Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.
# 
# IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

# In[58]:


for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Reviews'].str.contains(word)])))


# In[59]:


tf1


# The more the value of IDF, the more unique is the word.

# # Term Frequency – Inverse Document Frequency (TF-IDF)
# TF-IDF is the multiplication of the TF and IDF which we calculated above.

# In[60]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
data_vect = tfidf.fit_transform(data['Reviews'])

data_vect


# # Bag of Words
# Bag of Words (BoW) refers to the representation of text which describes the presence of words within the text data. The intuition behind this is that two similar text fields will contain similar kind of words, and will therefore have a similar bag of words. Further, that from the text alone we can learn something about the meaning of the document.
# 
# For implementation, sklearn provides a separate function for it as shown below

# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Reviews'])
data_bow


# # Sentiment Analysis
# If you recall, our problem was to detect the sentiment of the tweet. So, before applying any ML/DL models (which can have a separate feature detecting the sentiment using the textblob library), let’s check the sentiment of the first few tweets.

# In[63]:


data['Reviews'][:10].apply(lambda x: TextBlob(x).sentiment)


# # Here, we only extract polarity as it indicates the sentiment as value nearer to 1 means a positive sentiment and values nearer to -1 means a negative sentiment. This can also work as a feature for building a machine learning model.

# In[64]:


data['sentiment'] = data['Reviews'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Reviews','sentiment']].head()


# # Word Embeddings
# Word Embedding is the representation of text in the form of vectors. The underlying idea here is that similar words will have a minimum distance between their vectors.
# 
# Word2Vec models require a lot of text, so either we can train it on our training data or we can use the pre-trained word vectors developed by Google, Wiki, etc.

# In[ ]:





# In[65]:


get_ipython().system('pip install gensim')
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[95]:


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'negative-words.txt'
word2vec_output_file = 'positive-words.txtpd.read_csv'


# In[ ]:


#glove2word2vec(glove_input_file, word2vec_output_file)


# In[72]:



import collections
from collections import Counter
import nltk
nltk.download('punkt')


# In[73]:


from textblob import TextBlob
data['polarity'] = data['Reviews'].apply(lambda x: TextBlob(x).sentiment[0])
data[['Reviews','polarity']].head(5)


# In[74]:


# Displaying top 5 positive posts of Category_A
data[data.polarity>0].head(5)


# In[75]:


def sent_type(text): 
    for i in (text):
        if i>0:
            print('positive')
        elif i==0:
            print('neutral')
        else:
            print('negative')


# In[76]:


sent_type(data['polarity'])


# In[77]:


data["category"]=data['polarity']


# In[78]:


data.loc[data.category > 0,'category']="Positive"
data.loc[data.category !='Positive','category']="Negative"


# In[79]:


data["category"]=data["category"].astype('category')
data.dtypes


# In[80]:


sns.countplot(x='category',data=data,palette='hls')


# In[81]:


positive_reviews= data[data.category=='Positive']
negative_reviews= data[data.category=='Negative']
positive_reviews_text=" ".join(positive_reviews.Reviews.to_numpy().tolist())
negative_reviews_text=" ".join(negative_reviews.Reviews.to_numpy().tolist())
positive_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(positive_reviews_text)
negative_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(negative_reviews_text)
plt.imshow(positive_reviews_cloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()
plt.imshow(negative_reviews_cloud,interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()


# In[ ]:




