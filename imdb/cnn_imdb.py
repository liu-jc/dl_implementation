from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import *
from keras.utils.visualize_util import plot
from load_data import *
import numpy as np
import json

vocab_size = 20000
max_len = 200

(x_train,y_train),(x_test,y_test) = load_data("imdb_full.pkl",vocab_size=vocab_size)

#padding
print("padding")
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)
print("x_train.shape = %s" % str(x_train.shape))
print("x_test.shape = %s" % str(x_test.shape))

print("define model")
model = Sequential()
model.add(Embedding(vocab_size,128,dropout=0.5))
model.add(Convolution1D(nb_filter=256,filter_length=3,activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


print("train..")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
json_string = model.to_json()
open('cnn_model_architecture.json','w').write(json_string)

plot(model,to_file='cnn_model.png')

history = model.fit(x_train,y_train,nb_epoch=10,batch_size=32,validation_data=(x_test,y_test))
model.save_weights('cnn_model_weights.h5')



print("evaluate..")
score,acc = model.evaluate(x_test,y_test,batch_size=32)

print("test score:", score)
print("test accuracy:", acc)
print(history.history)
in_json = json.dumps(history.history)
open('history_cnn.json','w').write(history.history)
