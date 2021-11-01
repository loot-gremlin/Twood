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
