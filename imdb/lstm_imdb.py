from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import *
from load_data import *
import numpy as np

vocab_size = 20000
max_len = 100

(x_train,y_train),(x_test,y_test) = load_data("imdb_full.pkl",vocab_size=vocab_size)
#x_train = x_train[:10000]
#x_test = x_test[:10000]
#y_train = y_train[:10000]
#y_test = y_test[:10000]

#padding
print("padding")
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)
print("x_train.shape = %s" % str(x_train.shape))
print("x_test.shape = %s" % str(x_test.shape))

print("define model")
model = Sequential()
model.add(Embedding(vocab_size,128,dropout=0.2))
model.add(LSTM(128,dropout_W=0.2,dropout_U=0.2))
model.add(Dense(1,activation='sigmoid'))

print("train..")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,nb_epoch=5,batch_size=32,validation_data=(x_train,y_train))

print("evaluate..")
score,acc = model.evaluate(x_test,y_test,batch_size=32)

print("test score:", score)
print("test accuracy:", acc)
