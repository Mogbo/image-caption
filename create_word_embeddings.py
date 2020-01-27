import argparse
import numpy as np
import cPickle as pickle


parser = argparse.ArgumentParser()
parser.add_argument('--vectors_file', default='../glove.6B.300d.txt', type=str)
args = parser.parse_args()

#with open(args.vocab_file, 'r') as f:
#    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
with open(args.vectors_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = map(float, vals[1:])

embedding_dim = len(vals[1:])
with open('/scratch/costarendon.b/show-attend-and-tell-tensorflow/data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

embedding_matrix = np.empty((max(word_to_idx.values())+1, embedding_dim))

count = 0

for word, idx in word_to_idx.items():

    if word in vectors.keys():
        embedding_matrix[idx,:] = np.array(vectors[word])

    else:
        print("Word:", word, "not found")
        count = count + 1
        embedding_matrix[idx,:] = np.random.uniform(low=-1., high=1., size=embedding_dim)

print('Count of words not in glove:', count)
pickle.dump(embedding_matrix, open('embedding_matrix2.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)