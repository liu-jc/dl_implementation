import numpy as np
from six.moves import cPickle

def load_data(path,
            vocab_size = 1000,skip_top=0,maxlen=10000,
            start_char=1,oov_char=2, index_from=3):
    '''
    skip_top : skip the top N most frequently words
    maxlen : the sentence's maxlen
    start_char: mark the start of a sentence 
    oov_char: words that were cut out will be replaced with this char
    index_from: index actual words with this index.
    '''
    f = open(path,"rb")
    (x_train,label_train),(x_test,label_test) = cPickle.load(f)
    f.close()

    np.random.seed(113)
    np.random.shuffle(x_train)
    np.random.seed(113)
    np.random.shuffle(label_train)

    np.random.seed(113*2)
    np.random.shuffle(x_test)
    np.random.seed(113*2)
    np.random.shuffle(label_test)

    X = x_train + x_test
    labels = label_train + label_test
    
    #add start_char and change index 
    print("start_char")
    X = [[start_char] + [word + index_from for word in sentence] for sentence in X]

    #drop the sentence whoes len is larger than maxlen
    print("drop sentences")
    new_X = []
    new_labels = []
    for sentence,label in zip(X,labels):
        if len(sentence) <= maxlen:
            new_X.append(sentence)
            new_labels.append(label)
    X = new_X
    labels = new_labels

    
    # > vocab_size or < skip_top
    print("change words")
    X = [[oov_char if (word >= vocab_size or word < skip_top) else word for word in sentence] for sentence in X]

    print("change to nparray")

    x_train = np.asarray(X[:len(x_train)])
    x_test = np.array(X[len(x_train):])
    label_train = np.array(labels[:len(x_train)])
    label_test = np.array(labels[len(x_train):])
    print("done")

    return (x_train,label_train),(x_test,label_test)
    #return x_train,x_test,labels

if __name__ == '__main__':
    (x_train,label_train) ,(x_test,label_test) = load_data("imdb_full.pkl")
