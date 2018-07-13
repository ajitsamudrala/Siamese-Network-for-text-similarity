
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras import regularizers,initializers
from keras import models,layers,Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM,Dropout,Dense,Bidirectional,Embedding,BatchNormalization, Activation, Input, Conv1D,MaxPool1D,Flatten
from sklearn.model_selection import train_test_split
from Twitter_data_extraction import generate_training_data
from sklearn.model_selection import train_test_split
import time
import os


# In[6]:


train_len=8000
max_features=2000
max_len=180
embedding_dim=50


# In[3]:


data_1,data_2,labels,word_index,sequences=generate_training_data(train_len,max_features,max_len,'SenSanders','realDonaldTrump','HillaryClinton','SpeakerRyan','PressSec')


# In[12]:


file=open("C:/Users/ajit/Downloads/Compressed/glove50d.txt",encoding="utf8") #importing Glove vectors


# In[13]:


embeddings_index={}
for line in file:
    values=line.split()
    word=values[0]
    embeddings=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=embeddings
file.close()


# In[14]:


embedding_matrix=np.zeros((max_features,embedding_dim))
for word, i in word_index.items():
    if i < max_features:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# In[65]:


class Siamese_network():
    
    def __init__(self,embedding_dim,embedding_matrix,number_lstm_units,number_dense_units,max_len,max_features):
        
        self.embedding_dim=embedding_dim
        self.number_lstm_units=number_lstm_units
        self.number_dense_units=number_dense_units
        self.max_len=max_len
        self.max_features=max_features
        self.embedding_matrix=embedding_matrix

    def model(self,data_1,data_2,labels,model_save_directory='./'):
        
        reg=regularizers.l1(0.2)
        
        layer_1=Bidirectional(LSTM(self.number_lstm_units,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        layer_2=Bidirectional(LSTM(self.number_lstm_units,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        layer_3=Bidirectional(LSTM(self.number_lstm_units,return_sequences=False))
        embedding_layer=Embedding(self.max_features, self.embedding_dim, weights=[self.embedding_matrix],
                                    input_length=self.max_len, trainable=False)
        
        
        input_1=Input(shape=(self.max_len,), dtype='int32',name='input1')
        embedding_layer_1=embedding_layer(input_1)
        x_1=layer_1(embedding_layer_1)
        x_1=layer_2(x_1)
        x_1=layer_3(x_1)
        
        input_2=Input(shape=(self.max_len,), dtype='int32',name='input2')
        embedding_layer_2=embedding_layer(input_2)
        x_2=layer_1(embedding_layer_2)
        x_2=layer_2(x_2)
        x_2=layer_3(x_2)
        
        merged=concatenate([x_1,x_2],axis=-1)
        merged=BatchNormalization()(merged)
        merged=Dropout(0.1)(merged)
        merged=Dense(self.number_dense_units,kernel_regularizer=reg)(merged)
        merged=BatchNormalization()(merged)
        merged=Activation(activation='relu')(merged)
        output = Dense(1, activation='sigmoid')(merged)
        
        model=Model(inputs=[input_1,input_2],outputs=output)
        
        print(model.summary())
        
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        

        STAMP = 'lstm_%d_%d' % (self.number_lstm_units, self.number_dense_units)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([data_1, data_2], labels,
                  validation_split=0.3,
                  epochs=20, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path
        

