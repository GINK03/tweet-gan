import os
import sys
import math
import MeCab 
import glob
import json
import dill
import re
import pickle
filenum = len(glob.glob('./out/*'))
m = MeCab.Tagger('-Owakati')
def tag_counter():
  tag_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    if ni%1000 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
      pass
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    for tag in re.findall(r'#.*?\s', obj['txt']):
      #print(tag)
      if tag_freq.get(tag) is None: tag_freq[tag] = 0
      tag_freq[tag] += 1
  for tag, freq in sorted(tag_freq.items(), key=lambda x:x[1]*-1):
    print(tag, freq)
  

def cal_maxlen():
  maxlen = 0
  num_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    if ni%1000 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
      pass
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    wakati = m.parse(obj['txt']).strip().split()
    if num_freq.get(len(wakati)) is None: num_freq[len(wakati)] = 0
    num_freq[len(wakati)] += 1
  
  for num, freq in sorted(num_freq.items(), key=lambda x:x[0]):
    print( "{} {}".format(num, freq))
  open('maxlen.txt', 'w').write(str(max(num_freq.keys())))
  # 214個がmaxだった
  # なんか30語で十分だわ
def build_wakati_texts():
  maxlen = 25
  padding = 5
  term_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    buff = ['*']*30
    padding = ['*']*4
    if ni%10 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    wakati = m.parse(obj['txt']).strip().split()
    for i, term in enumerate(wakati):
      try:
        buff[i] = term
      except IndexError as e:
        break
    padding.extend(buff)
    print(' '.join(padding) )

def vectorizer():
  term_vec = {}
  with open('./model.vec', 'r') as f:
    next(f)
    for i, line in enumerate(f):
      if i%1000 == 0:
        print("now iter %d "%(i))
      ents = iter(line.strip().split())
      term = next(ents)
      vec  = list(map(float, ents))
      term_vec[term] = vec 

  open('term_vec.pkl', 'wb').write(pickle.dumps(term_vec) )
      
if __name__ == '__main__':
  #tag_counter()
  #cal_maxlen()
  #build_wakati_texts()
  vectorizer()
  
