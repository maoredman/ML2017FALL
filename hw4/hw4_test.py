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

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

model_name = 'models/cbow_GRU_128batch-02-0.82.hdf5'
loaded_model = load_model(model_name)
print('model loaded!')

x_submission = []

with open(sys.argv[1], 'rt') as testfile:
    reader = csv.reader(testfile, delimiter=',')
    next(reader) # skip headings
    for row in reader:
        # print(''.join(row[1:]))
        words = []
        # print(''.join(row[1:]).split())
        for word in ''.join(row[1:]).split():
            try:
                words.append(vocab[word])
            except:
                words.append(0) ############ 0 actually corresponds to 'i'
        x_submission.append(words)

max_review_length = 40
x_submission = sequence.pad_sequences(x_submission, maxlen=max_review_length)
print('finished generating test data!')

preds = loaded_model.predict(x_submission, batch_size=1024, verbose=1)
print('generated predictions!')

outname = sys.argv[2]
if not os.path.exists(os.path.dirname(outname)):
    try:
        os.makedirs(os.path.dirname(outname))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(outname, 'wt') as outfile:
    test_writer = csv.writer(outfile)
    test_writer.writerow(['id','label'])   
    counter = 0
    for i in (preds > 0.5):
        test_writer.writerow([counter, int(i)])
        counter += 1  
print('finished writing submission!')