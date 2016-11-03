import csv
import os
import numpy as np

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_data(filename):
    
    with open(filename) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)
        feature_names = np.array(temp)
        
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype = np.float)
            target[i] = np.asarray(d[-1], dtype = np.float)
            
    return Bunch(sample = n_samples, features = n_features, data = data, target = target, feature_names = feature_names)

def write_data(filename, layerNum, list):
    sampleData = load_data(os.getcwd() + '/static/uploadfiles/' + filename)
    writer = csv.writer(file(os.getcwd() + '/static/results/' + 'data-result' + str(layerNum) + '.csv', 'w'))
    writer.writerow([sampleData.sample, len(list)])
    writer.writerow(sampleData.feature_names[list + [-1]])
    for i in range(len(sampleData.data)):
        writer.writerow(np.append(sampleData.data[i], sampleData.target[i])[list + [-1]])


if __name__ == '__main__':
    Sample_data = load_data("data1.csv")
    write_data('data1.csv', 2, [0, 2])
