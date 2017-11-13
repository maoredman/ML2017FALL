from keras.models import load_model
import csv
import sys
import os
import errno
import numpy as np

x_submission = []

with open(sys.argv[1], 'rt', encoding='big5') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    row1 = next(reader) # skip headings
    for idx, row in enumerate(reader):
        img_data = [int(i) for i in row[1].split(' ')]
        x_submission.append(img_data)

x_submission = np.array(x_submission)
x_submission = x_submission.reshape(x_submission.shape[0], 48, 48, 1)
x_submission = x_submission.astype('float32')
x_submission /= 255
print('test data loaded!')

loaded_model = load_model('200.hdf5')
pred = loaded_model.predict(x_submission, batch_size=1)
pred_to_category_num = [np.argmax(i) for i in pred]
print('generated test predictions!')

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
    for num in pred_to_category_num:
        test_writer.writerow([counter, num])
        counter += 1
    
print('finished writing submission!')
