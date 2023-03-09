# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:01:38 2021

@author: hevc
"""

## Importing Libraries
import numpy as np
import warnings
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dropout, Embedding, SpatialDropout1D, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from nltk.tokenize import word_tokenize
import datetime
from stop_words import get_stop_words
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

#pretrained embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('C:/Users/hevc/Documents/mareeta/coen342/project2/glove.6B.200d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions

glove_file.close()

#hyperparameters

embedding_dims = 64
max_words= 5000
OOV_TOK = '<OOV>'
trunc_type = 'post'
padding_type = 'post'

warnings.filterwarnings('ignore')
## getting the current path
cur_dir = os.getcwd()

# Iterating the contents of the file
x_train = open(os.path.join(cur_dir,"train.txt"), encoding="utf8").readlines()
x_test = open(os.path.join(cur_dir,"test.txt"), encoding="utf8").readlines()
x_val = open(os.path.join(cur_dir,"val.txt"), encoding="utf8").readlines()
y_train = open(os.path.join(cur_dir,"train.labels"), encoding="utf8").readlines()
y_val = open(os.path.join(cur_dir,"val.labels"), encoding="utf8").readlines()

#cleaning text
stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop_words.extend(nltk_words)

x_train = [w for w in x_train if not w in stop_words]
x_val = [w for w in x_val if not w in stop_words]
x_test = [w for w in x_test if not w in stop_words]

#tokenizer-training data
tok = Tokenizer(num_words=max_words,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tok.fit_on_texts(x_train)
word_index = tok.word_index
print('Vocabulary size:', len(word_index))
vocab_length = len(word_index)+1

sequences = tok.texts_to_sequences(x_train)

#padding
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(x_train, key=word_count)
maxlen = len(word_tokenize(longest_sentence))

print('Pad sequences (samples training)...')
sequences_train = sequence.pad_sequences(sequences, maxlen=maxlen,padding=padding_type, truncating=trunc_type)

#tokenizer/padding-validation data
sequences = tok.texts_to_sequences(x_val)
print('Pad sequences (samples validating)...')
sequences_val = sequence.pad_sequences(sequences, maxlen=maxlen,padding=padding_type, truncating=trunc_type)

#pretrained embeddings
embedding_matrix = zeros((vocab_length, embedding_dims))
for word, index in tok.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
#converting labels to np array
training_label_seq = np.array(y_train,dtype=np.uint8)
validation_label_seq = np.array(y_val, dtype=np.uint8)

#checking whether decoding is right
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#print(decode_article(sequences_train[11]))

#gradient-clipping
#optimizer = optimizers.Adam(clipvalue=1, lr=1e-9)
print('Build model...')
model = Sequential()
model.add(Embedding(vocab_length, embedding_dims,input_length=maxlen,mask_zero=True))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(embedding_dims,dropout=0.2,return_sequences=True)))
model.add(Bidirectional(LSTM(embedding_dims,dropout=0.2,return_sequences=True)))
model.add(Bidirectional(LSTM(embedding_dims,dropout=0.2)))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
print(model.summary())


epochs = 30
batch_size = 64

#history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

print('Train...')
logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=logdir)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
history = model.fit(sequences_train, training_label_seq,batch_size=batch_size,epochs=epochs,validation_data=(sequences_val, validation_label_seq),callbacks=[tensorboard_callback],)

print('Test...')
test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=maxlen,padding=padding_type, truncating=trunc_type)
result = model.predict(test_sequences_matrix)
shape=(5227,1)
prediction = np.zeros(shape)
i = 0
for i in range(0,5227):
    prediction[i] = np.argmax(result[i])
    i = i+1

np.savetxt("results.txt",prediction,fmt="%d")

    
    
    
    
