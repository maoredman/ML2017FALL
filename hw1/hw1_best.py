
import numpy as np
import csv
import sys
import os
import errno

weights = np.loadtxt('best_weights.txt')
# weights = []
# with open('objs.pickle') as f:
#	weights = pickle.load(f)

test_AMB_TEMP = []
test_CH4 = []
test_CO = []
test_NMHC = []
test_NO = []
test_NO2 = []
test_NOx = []
test_O3 = []
test_PM10 = []
test_PM2_5 = []
test_RAINFALL = []
test_RH = []
test_SO2 = []
test_THC = []
test_WD_HR = []
test_WIND_DIREC = []
test_WIND_SPEED = []
test_WS_HR = []

window = 8

with open(sys.argv[1], 'rt', encoding='big5') as infile:
    outname = sys.argv[2]
    if not os.path.exists(os.path.dirname(outname)):
        try:
            os.makedirs(os.path.dirname(outname))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(outname, 'wt') as outfile:
        test_reader = csv.reader(infile, delimiter='\n')
        test_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['id','value'])
        counter = 0
        for idx, row in enumerate(test_reader):
            
            row = row[0].split(',')
            # print('row', row)
            values = row[2:] # remove id, category
            values = [float(i) if i != 'NR' else 0.0 for i in values]
            
            category_num = idx % 18
            if category_num == 0:
                test_AMB_TEMP = values
            elif category_num == 1:
                test_CH4 = values
            elif category_num == 2:
                test_CO = values
            elif category_num == 3:
                test_NMHC = values
            elif category_num == 4:
                test_NO = values
            elif category_num == 5:
                test_NO2 = values
            elif category_num == 6:
                test_NOx = values
            elif category_num == 7:
                test_O3 = values
            elif category_num == 8:
                test_PM10 = values
            elif category_num == 9:
                test_PM2_5 = values
            elif category_num == 10:
                test_RAINFALL = values
            elif category_num == 11:
                test_RH = values
            elif category_num == 12:
                test_SO2 = values
            elif category_num == 13:
                test_THC = values
            elif category_num == 14:
                test_WD_HR = values
            elif category_num == 15:
                test_WIND_DIREC = values
            elif category_num == 16:
                test_WIND_SPEED = values
            elif category_num == 17:
                test_WS_HR = values
                
                # finished reading an id, can go to work now
                features = np.array(test_PM2_5[-window:])
                features = np.append(features, test_RAINFALL[-window:])
                # features = np.append(features, NMHC[idx:idx+window])
                features = np.append(features, test_PM10[-window:])
                features = np.append(features, test_O3[-window:])
                # features = np.append(features, np.array(PM10[idx:idx+window])**2)
                features = np.append(features, np.array(test_RAINFALL[-window:])**3)
                features = np.append(features, test_NOx[-window:])
                # features = np.append(features, np.array(PM2_5[idx+window-1])**2)
                features = np.append(features, np.array(test_PM2_5[-window:]) * np.array(test_PM10[-window:]))
                features = np.append(features, test_WIND_DIREC[-3:])
                features = np.append(features, np.array(test_WIND_DIREC[-3:]) ** 2)
                features = np.append(features, test_CH4[-window:])
                features = np.append(features, np.array(test_CH4[-window:])**2)
                features = np.append(features, np.array(test_WIND_DIREC[-window:])* np.array(test_PM2_5[-window:]))
                features = np.append(features, test_WIND_SPEED[-3:])
                features = np.append(features, test_RH[-window:])
                features = np.append(features, np.array(test_RH[-window:])**2)
                features = np.append(features, test_SO2[-window:])
                features = np.append(features, np.array(test_SO2[-window:])**2)
                features = np.append(features, [1.0])

                prediction = sum(np.multiply(weights, features))
                
                if prediction < 0:
                    prediction = 0.0
                
                test_writer.writerow(['id_' + str(counter),prediction])
                counter += 1

print('done!')