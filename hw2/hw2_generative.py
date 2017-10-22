import numpy as np
import math
import csv
import pickle
import sys
import os
import errno

with open('gen_vars.pickle', 'rb') as infile:
    prob0, prob1, mu_0, mu_1, cov_0, cov_1, prob0_0, prob0_1 = pickle.load(infile)


with open(sys.argv[5], 'rt', encoding='big5') as infile:
    reader = csv.reader(infile)
    row1 = next(reader) # skip headings

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
        
        count = 0
        for row in reader:
            X_input = np.array([float(i) for i in row])
            X_input_cont = X_input[[0,1,3,4,5]]
            X_input_disc = np.delete(X_input, [0,1,3,4,5])
            
            # print('here ok 1')
            
            # np.linalg.inv
            # np.linalg.pinv
            X_cont_in_0 = 1 / (pow(2*math.pi, len(X_input_cont)/2) + pow(10,-8)) / (pow(np.linalg.det(cov_0),0.5) + pow(10,-8)) \
            * math.exp(-0.5*(X_input_cont - mu_0).T.dot(np.linalg.inv(cov_0)).dot(X_input_cont - mu_0))
            
            X_cont_in_1 = 1 / (pow(2*math.pi, len(X_input_cont)/2) + pow(10,-8)) / (pow(np.linalg.det(cov_1),0.5) + pow(10,-8)) \
            * math.exp(-0.5*(X_input_cont - mu_1).T.dot(np.linalg.inv(cov_1)).dot(X_input_cont - mu_1))
            
            # print('here ok 2')
            
            X_input_disc_0 = np.delete(X_input, [0,1,3,4,5])
            X_input_disc_1 = np.delete(X_input, [0,1,3,4,5])
            for idx in range(0, len(X_input_disc)):
                if(X_input_disc[idx] == 0):
                    X_input_disc_0[idx] = prob0_0[idx]
                    X_input_disc_1[idx] = prob0_1[idx]
                else:
                    X_input_disc_0[idx] = 1 - prob0_0[idx]
                    X_input_disc_1[idx] = 1 - prob0_1[idx]
            # multiply all the probabilities together
            X_disc_in_0 = X_input_disc_0.prod()
            X_disc_in_1 = X_input_disc_1.prod()
            
            prob_X_in_0 = X_disc_in_0 * X_cont_in_0
            prob_X_in_1 = X_disc_in_1 * X_cont_in_1
            
            prob_0_given_x = prob_X_in_0 * prob0 / (prob_X_in_0 * prob0 + prob_X_in_1 * prob1)
            
            if np.isnan(prob_0_given_x):
                print('prob', prob_0_given_x)
                print(X_input_disc)
                print(np.delete(X_input, [0,1,3,4,5]))
                # print(np.delete(X_input, [0,1,3,4,5]))
                break
                
            prediction = 1
            if prob_0_given_x > 0.5:
                prediction = 0
            
            count += 1
            test_writer.writerow([str(count), prediction])
            
print('done!')