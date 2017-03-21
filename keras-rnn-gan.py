from __future__ import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Dense, Activation
from keras.layers               import LSTM, GRU, SimpleRNN
from keras.optimizers           import RMSprop, Adam
from keras.utils.data_utils     import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.noise         import GaussianDropout as GD
import numpy as np
import random
import sys
import tensorflow               as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import json
import pickle
import msgpack
import msgpack_numpy as mn
mn.patch()
import MeCab
import plyvel
from itertools import cycle as Cycle
import dill

def build_model(maxlen=None, out_dim=None, in_dim=256):
  print('Build model...')
  model = Sequential()
  model.add(GRU(128*25, return_sequences=False, input_shape=(maxlen, in_dim)))
  model.add(BN())
  model.add(Dense(out_dim))
  model.add(Activation('linear'))
  optimizer = Adam()
  model.compile(loss='mse', optimizer=optimizer) 
  return model

def preexe():
  """ ./fasttext skipgram -input dumps.txt -output model -dim 256 -minCount 1
  """
  from collections import OrderedDict as odict
  term_index = odict()
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  with open('./dumps.txt', 'r') as f:
    datasets = []
    for fi, line in enumerate(f):
      if fi > 50000: break
      if fi%500 == 0:
        print("now iter {}".format(fi))
      terms = line.strip().split()
      for slide in range(0, len(terms) - 4, 1 ):
        ans  = terms[slide+4] 
        buff = []
        try:
          [buff.append(term_vec[term]) for term in terms[slide: slide+4]]
        except KeyError as e:
          continue
        datasets.append( (buff, ans, terms[slide: slide+5]) )
        if term_index.get(ans) is None:
          term_index[ans] = len(term_index)
  open('datasets.pkl', 'wb').write(pickle.dumps(datasets))
  open('term_index.pkl', 'wb').write(pickle.dumps(term_index))

def train():
  print("importing data from serialized...")
  datasets    = pickle.loads(open('datasets.pkl', 'rb').read())
  term_index  = pickle.loads(open('term_index.pkl', 'rb').read())
  term_vec    = pickle.loads(open('term_vec.pkl', 'rb').read())
  print("unpacking is done...")
  sentences = []
  answers   = []
  counter = 0
  for dbi, series in enumerate(datasets):
    # 64GByteで最大80万データ・セットくらいまで行ける
    if counter > 1000000: break
    vec, ans, text = series 
    if all(list(map(lambda x:x=="*", text))):
      #print(text)
      continue
    counter += 1
    sentences.append(np.array(vec))
    answers.append(term_vec[ans])
  print('nb sequences:', len(sentences))

  print('Vectorization...')
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.float64)
  y = np.zeros((len(sentences), 256), dtype=np.float64)
  for i, sentence in enumerate(sentences):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, vec in enumerate(sentence):
      X[i, t, :] = vec
    y[i, :] = answers[i]
  model    = build_model(maxlen=len(sentences[0]), in_dim=256, out_dim=256)
  for iteration in range(1, 101):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    MODEL_NAME = "./models/snapshot.%09d.model"%(iteration)
    model.save(MODEL_NAME)
  sys.exit()

def pred():
  from scipy import linalg, mat, dot
  print("start to loading term_vec")
  term_vec    = pickle.loads(open('term_vec.pkl', 'rb').read())
  term_norm   = {}
  for term, vec in term_vec.items():
    term_vec[term] = vec
    term_norm[term] = linalg.norm(vec)
    #print(  linalg.norm(vec) )
  model_type = sorted(glob.glob('./models/snapshot.*.model'))[-1]
  print("model type is %s"%model_type)
  model  = load_model(model_type)
  datasets    = pickle.loads(open('datasets.pkl', 'rb').read())
  for dataset in datasets:
    vec, ans, text = dataset
    #print(vec)
    #print(ans)
    X = np.array([vec])
    results = model.predict(X)
    #for res in results:
    res = results[0]
    #print(res)
    res = results[0].tolist()
    human = sorted([(term, float(dot(vec, res)/term_norm[term]/linalg.norm(res)) ) \
            for term, vec in term_vec.items()], key=lambda x:x[1]*-1)
    print(res)
    print(text)
    print(human[0])


def main():
  if '--preexe' in sys.argv:
     preexe()
  if '--train' in sys.argv:
     train()
  if '--pred' in sys.argv:
     pred()
if __name__ == '__main__':
  main()
