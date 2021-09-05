import pandas as pd
import numpy as np
import os

def GetNames(filepath):
    return filepath.split('\\')[-1]

def ImportData(path):
    colums = ['centre' , 'left' , 'right' , 'steering' , 'throttle' , 'break' , 'speed']
    data = pd.read_csv(os.path.join(path , 'driving_log.csv') , names=colums)
    data['centre'] = data['centre'].apply(GetNames)
    data['left'] = data['left'].apply(GetNames)
    data['right'] = data['right'].apply(GetNames)
    return data

def ConvertDataPath(path , data):
    ImagePath = []
    Steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        ImagePath.append(os.path.join(path , 'IMG' , indexedData[0]))
        Steering.append(float(indexedData[3]))

    ImagePath = ImagePath
    Steering = np.asarray(Steering)
    return ImagePath,Steering

