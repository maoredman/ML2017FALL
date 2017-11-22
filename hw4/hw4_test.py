import numpy as np
import csv
import sys
import os
import errno
import pickle
from keras.models import load_model
from keras.preprocessing import sequence

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model_name = 'models/03-0.80.hdf5'
loaded_model = load_model(model_name)
print('model loaded!')

x_submission = []

with open(sys.argv[1], 'rt') as testfile:
    reader = csv.reader(testfile, delimiter=',')
    next(reader) # skip headings
    for row in reader:
        x_submission.append(''.join(row[1:]))

max_review_length = 40
x_submission = tokenizer.texts_to_sequences(x_submission)   
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