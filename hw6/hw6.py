import numpy as np
import csv
import sys
import os
import errno
import pickle

"""with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)"""

with open('labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)


x_submission = []
with open(sys.argv[2], 'rt') as testfile:
    reader = csv.reader(testfile, delimiter=',')
    next(reader) # skip headings
    for row in reader:
        # print([int(row[1]), int(row[2])])
        x_submission.append([int(row[1]), int(row[2])])
print('finished reading file')


predictions = []
for id_pair in x_submission:
    if labels[id_pair[0]] == labels[id_pair[1]]:
        predictions.append(1)
    else:
        predictions.append(0)


outname = sys.argv[3]
if not os.path.exists(os.path.dirname(outname)):
    try:
        os.makedirs(os.path.dirname(outname))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(outname, 'wt') as outfile:
    test_writer = csv.writer(outfile)
    test_writer.writerow(['ID','Ans'])
    
    counter = 0
    for i in predictions:
        test_writer.writerow([counter, int(i)])
        counter += 1
    
print('finished writing submission!')
