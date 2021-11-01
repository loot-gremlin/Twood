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
#print(partial_train.shape, partial_train[:4])
'''
embeddings_index = {}
f = open('glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
'''
embedding_dim = 100
'''
embedding_matrix = np.zeros((WORDS, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < WORDS:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
'''
model = models.Sequential()
#print(np.random.randint(10000, size=(32,10)))
model.add(layers.Embedding(WORDS, embedding_dim, input_length=len(partial_train[0])))
model.add(layers.Bidirectional(layers.GRU(64, dropout=0.3, recurrent_dropout=0.35)))
#model.add(layers.GRU(64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
#model.add(layers.GRU(128, activation='relu', dropout=0.1, recurrent_dropout=0.5))
#model.add(layers.LSTM(64))
#model.add(layers.Flatten())
#model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Conv1D(32, 7, activation='relu'))
#model.add(layers.MaxPooling1D(5))
#model.add(layers.Conv1D(32, 5, activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
'''
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
'''
print(model.summary())
cbs = [
    keras.callbacks.ModelCheckpoint("model{epoch:02d}-{val_mean_absolute_error:.2f}.h5", monitor='val_mean_absolute_error', verbose=0, save_best_only=True)
]
#model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(partial_train, partial_label, epochs=40, batch_size=64, callbacks=cbs, validation_data=(val_data, val_label))
average_mae_history = history.history['val_mean_absolute_error']
a = plt.figure(1)
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
b = plt.figure(2)
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history, 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

a.show()
b.show()


plt.show()
