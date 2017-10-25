# from keras.models import load_model
import csv
import sys
import os
import errno
import numpy as np
import xgboost as xgb
import pandas as pd
import pickle

final_gb = xgb.Booster({'nthread': 4})  # init model
final_gb.load_model('xgboost.model')

# with open('xgboost_model.pickle', 'rb') as handle:
    # final_gb = pickle.load(handle)

test_set = pd.read_csv(sys.argv[2], skiprows = 1, header = None)
# print(test_set.head())
col_labels2 = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
test_set.columns = col_labels2
for feature in test_set.columns: # Loop through all columns in the dataframe
    if test_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        test_set[feature] = pd.Categorical(test_set[feature]).codes # Replace strings with an integer

final_test = test_set
testdmat = xgb.DMatrix(final_test)
y_pred = final_gb.predict(testdmat)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

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
    for num in y_pred:
        counter += 1
        test_writer.writerow([str(counter),int(num)])

print('done!')

'''model = load_model('nn_model.h5')

X_test = []
with open(sys.argv[5], 'rt', encoding='big5') as infile:
    reader = csv.reader(infile)

    row1 = next(reader) # skip headings
    for row in reader:
        X_test.append([float(i) for i in row] + [1.0])

X_test = np.array(X_test)
for col in [0,1,3,4,5]:
    if np.std(X_test[:,col]) != 0:
        X_test[:,col] = np.divide((X_test[:,col] - np.average(X_test[:,col])), np.std(X_test[:,col]))

pred = model.predict(X_test, batch_size=1)
pred = np.around(pred).flatten()
# print(pred[0:100])

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

print('done!')'''