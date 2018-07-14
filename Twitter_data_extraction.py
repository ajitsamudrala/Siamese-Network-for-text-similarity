
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tweepy
from tweepy import OAuthHandler
import csv
import re




# In[107]:


def get_all_tweets(screen_name):
   
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    alltweets = []  

   
    new_tweets = api.user_timeline(screen_name = screen_name,count=200,tweet_mode='extended')

    
    alltweets.extend(new_tweets)

    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest,tweet_mode='extended')
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

    
    alltweets=[tweet.full_text for tweet in alltweets]
    cleaned_tweets=alltweets.copy()   
    
    for tweet in alltweets:
        if re.search("^RT\s",tweet):
            cleaned_tweets.remove(tweet)
   
    
    cleaned_tweets=[re.sub('https[^\s]+','',tweet) for tweet in cleaned_tweets]
    cleaned_tweets=[re.sub('(#|@|\d)[^\s]+\s?','',tweet) for tweet in cleaned_tweets]
    cleaned_tweets=[tweet.strip() for tweet in cleaned_tweets]
      
    return cleaned_tweets


# In[108]:


def retrieve_tweets(*args):
    alltweets=[]
    lengths=[]
    labels=[]
    for label,person in enumerate(args):
        print("Retrieving {}'s tweets".format(person))
        cleaned_tweets=get_all_tweets(person)
        alltweets.extend(cleaned_tweets)
        lengths.append(len(cleaned_tweets))
        labels.append(label)
    return alltweets,lengths,labels


# In[109]:


def texts_to_sequences(max_features,max_len,*args):
    alltweets,lengths,labels=retrieve_tweets(*args)
    tokenizer=Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(alltweets)
    sequences=tokenizer.texts_to_sequences(alltweets)
    sequences=sequence.pad_sequences(sequences,maxlen=max_len)
    return sequences,lengths,labels,tokenizer.word_index


# In[110]:


def generate_random_labels(equal_or_unequal,num_of_labels):
    random_label_1=np.random.randint(num_of_labels)
    random_label_2=np.random.randint(num_of_labels)
    if equal_or_unequal=='equal':
        if random_label_1==random_label_2:
            return random_label_1,random_label_2
        else:
            return(generate_random_labels(equal_or_unequal,num_of_labels))
    else:
        if random_label_1==random_label_2:
            return(generate_random_labels(equal_or_unequal,num_of_labels))
        else:
            return random_label_1,random_label_2


# In[111]:


def generate_indicies(random_label_1,random_label_2,lengths):
    index_1=np.random.randint(sum(lengths[:random_label_1]),sum(lengths[:random_label_1+1]))
    index_2=np.random.randint(sum(lengths[:random_label_2]),sum(lengths[:random_label_2+1]))
    if index_1==index_2:
        index_1,index_2=generate_indicies(random_label_1,random_label_2,lengths)
    return index_1,index_2


# In[121]:


def generate_training_data(train_len,max_features,max_len,*args):
    data_1=[]
    data_2=[]
    data_labels=[]
    sequences,lengths,labels,word_index=texts_to_sequences(max_features,max_len,*args)
    for i in range(round(train_len/2)):
        random_label_1,random_label_2=generate_random_labels('equal',len(np.unique(labels)))
        index_1,index_2=generate_indicies(random_label_1,random_label_2,lengths)
        data_1.append(sequences[index_1])
        data_2.append(sequences[index_2])
        data_labels.append(1)
    for i in range(round(train_len/2)):
        random_label_1,random_label_2=generate_random_labels('unequal',len(np.unique(labels)))
        index_1,index_2=generate_indicies(random_label_1,random_label_2,lengths)
        data_1.append(sequences[index_1])
        data_2.append(sequences[index_2])
        data_labels.append(0)
    shuffle_indices = np.random.permutation(np.arange(len(data_1)))
    data_1=np.array(data_1)[shuffle_indices]
    data_2=np.array(data_2)[shuffle_indices]
    data_labels=np.array(data_labels)[shuffle_indices]
    return data_1,data_2,data_labels,word_index,sequences

