import numpy as np
import csv
import sys
import os
import errno
import pickle
from keras.models import load_model
from keras.preprocessing import sequence

"""with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)"""

with open('user_tokenizer.pickle', 'rb') as handle:
    user_tokenizer = pickle.load(handle)

with open('movie_tokenizer.pickle', 'rb') as handle:
    movie_tokenizer = pickle.load(handle)

test_user_ids = []
test_movie_ids = []

with open(sys.argv[1], 'rt') as testfile:
    reader = csv.reader(testfile, delimiter=',')
    next(reader) # skip headings
    for row in reader:
        test_user_ids.append(row[1])
        test_movie_ids.append(row[2])
print('finished reading file')

test_user_tokens = [i if i != [] else [0] for i in user_tokenizer.texts_to_sequences(test_user_ids)]
test_movie_tokens = [i if i != [] else [0] for i in movie_tokenizer.texts_to_sequences(test_movie_ids)]

loaded_models = []
model_names = os.listdir('for_ensemble')
for model_name in model_names:
    loaded_model = load_model('for_ensemble/' + model_name)
    loaded_models.append(loaded_model)
print(len(loaded_models), 'models loaded!')

preds = np.array([[i] for i in np.zeros(len(test_user_tokens))])
for loaded_model in loaded_models:
    single_preds = loaded_model.predict([np.array(test_user_tokens), np.array(test_movie_tokens),
               np.zeros((len(test_user_tokens), 1)), np.zeros((len(test_movie_tokens), 1))], batch_size=500, verbose=1)
    preds += single_preds*5 / len(loaded_models)
print('ensemble prediction with', len(loaded_models), 'models completed!')

outname = sys.argv[2]
if not os.path.exists(os.path.dirname(outname)):
    try:
        os.makedirs(os.path.dirname(outname))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(outname, 'wt') as outfile:
    test_writer = csv.writer(outfile)
    test_writer.writerow(['TestDataID','Rating'])
    
    counter = 1
    for i in preds:
        test_writer.writerow([counter, i[0]])
        counter += 1
    
print('finished writing submission!')
