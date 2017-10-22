import numpy as np
import csv
import sys
import os
import errno

weights = np.load('weights_logistic.npy')

X_test = []

def sigmoid(x): # this sigmoid works with numpy arrays
    return 1.0 / (1 + np.exp(-x))

with open(sys.argv[5], 'rt', encoding='big5') as infile:
    reader = csv.reader(infile)

    row1 = next(reader) # skip headings
    for row in reader:
        X_test.append([float(i) for i in row] + [1.0])
            
X_test = np.array(X_test)
for col in [0,1,3,4,5]:
    if np.std(X_test[:,col]) != 0:
        X_test[:,col] = np.divide((X_test[:,col] - np.average(X_test[:,col])), np.std(X_test[:,col]))

pred = np.dot(X_test, weights)
pred = sigmoid(pred)
pred = np.around(pred)


outname = sys.argv[6]
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
    for num in pred:
        counter += 1
        test_writer.writerow([str(counter),int(num)])


print('done!')