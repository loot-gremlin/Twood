import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import save_model
import pickle


def every_day_im_shuffling(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

SIZE = 100000
sentences = []
counter = 0
ones = 0
labels = np.zeros(shape=SIZE, dtype=np.int8)
with open("fulldata.csv") as f:
    for line in f:
        if (counter >= SIZE):
            break
        l = line.split(',')
        #print(len(l))
        #print(l)
        if (ones > SIZE//2 and int(l[0][1]) == 0):
            continue
        labels[counter] = int(l[0][1])
        if (labels[counter] == 0):
            ones += 1
        sentences.append(l[5])
        counter += 1
print(np.average(labels))
print(sentences[0:10])
WORDS = 10000
tokenizer = Tokenizer(num_words=WORDS)
tokenizer.fit_on_texts(sentences)
seqeunces = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
print(len(seqeunces[0]), len(seqeunces), len(sentences))
data = pad_sequences(seqeunces, padding='post')
print(data[:4], len(data[0]))
print(labels[:4])
