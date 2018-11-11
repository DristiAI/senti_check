import keras
from keras.models import Model
from keras.layers import Conv1D,Dense,Flatten,Concatenate,BatchNormalization,Input,Embedding
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflowjs as tfjs

import numpy as np
import pandas as pd


TRAIN_FILE = '../train.csv'
TEST_FILE = '../test.csv'

#initialize word tokenizer
TOKENIZER = Tokenizer(num_words=10000)
                     
def datify(text_file):
    """
    input : text file containing data

    outputs : 
    tokenized sentences
    target sentiment list
    max length of the tokenized sentences

    """

    sentences=[]
    sentiments=[]
    maxlength =0

    with open(text_file,errors='ignore') as f:
        
        #parse first line
        f.readline()

        for line in f:
            
            try:
                _,sentiment,sentence = line.split(',')
                sentences.append(sentence)
                len_sent = len(sentence)
                if(maxlength<len_sent):
                    maxlength = len_sent
                sentiments.append(sentiment)
            except:
                pass

    #fit the tokenizer on sentences 
    TOKENIZER.fit_on_texts(sentences)
    #convert the sentences into sequences of numbers
    sentences = TOKENIZER.texts_to_sequences(sentences)
    #pad the sentences to equal length
    sentences = pad_sequences(sentences,maxlen = maxlength)

    return sentences,sentiments,maxlength

train_x,train_y,train_length = datify(TRAIN_FILE)

word_index = TOKENIZER.word_index

#create mappings csv file 
with open('mappings.csv','w') as f:
    i=0
    for key,value in word_index.items():
        if i == 10000:
            break
        f.write(str(key))
        f.write(',')
        f.write(str(value))
        f.write('\n')
        i+=1

vocab_size = len(TOKENIZER.word_index)+1

#define layers
input1 = Input(shape=(train_length,))
embedding1 = Embedding(vocab_size,100)(input1)
layer1_group1_conv = Conv1D(filters=32,kernel_size=2,activation='relu')(embedding1)
layer1_group1_bn = BatchNormalization()(layer1_group1_conv)
layer1_group1_flat = Flatten()(layer1_group1_bn)

input2 = Input(shape=(train_length,))
embedding2 = Embedding(vocab_size,100)(input2)
layer1_group2_conv = Conv1D(filters=32,kernel_size=3,activation='relu')(embedding2)
layer1_group2_bn = BatchNormalization()(layer1_group2_conv)
layer1_group2_flat = Flatten()(layer1_group2_bn)

input3 = Input(shape=(train_length,))
embedding3 = Embedding(vocab_size,100)(input3)
layer1_group3_conv = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding3)
layer1_group3_bn = BatchNormalization()(layer1_group3_conv)
layer1_group3_flat = Flatten()(layer1_group3_bn)

layer2_concat = Concatenate()([layer1_group1_flat,layer1_group2_flat,layer1_group3_flat])

layer3_dense = Dense(10,activation='relu')(layer2_concat)

layer4_out = Dense(1,activation='sigmoid')(layer3_dense)

model = Model(inputs=[input1,input2,input3],outputs=layer4_out)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#plot the model structure 
plot_model(model,to_file='../images/model.png',show_shapes=True)

#keras model accepts numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

model.fit([train_x,train_x,train_x],train_y,epochs=8,batch_size=32) 

model.save('../model/model.h5')

#converting to tensorflowjs model format
tfjs.converters.save_keras_model('../model')
