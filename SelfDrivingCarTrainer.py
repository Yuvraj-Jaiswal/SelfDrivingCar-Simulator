import TrainingFunctions
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import cv2
import numpy as np

def ConverPathtoImgArray(TrainX,TestX):
    Train_Img_Array = []
    Test_Img_Array = []

    for Train_path_img in TrainX:
        img = cv2.imread(Train_path_img)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = img[60:160,0:360]
        img = np.asarray(img)
        Train_Img_Array.append(img)

    for Test_path_img in TestX:
        img = cv2.imread(Test_path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[60:160, 0:360]
        img = np.asarray(img)
        Test_Img_Array.append(img)

    return Train_Img_Array,Test_Img_Array
def LoadModel():
    Model = Sequential()
    Model.add(Conv2D(input_shape=(100, 320, 3), kernel_size=(5, 5), filters=32, activation='elu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(kernel_size=(5, 5), filters=32, activation='elu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(kernel_size=(5, 5), filters=32, activation='elu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(kernel_size=(5, 5), filters=64, activation='elu'))
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Flatten())
    Model.add(Dense(100, activation='elu'))
    Model.add(Dense(50, activation='elu'))
    Model.add(Dense(25, activation='elu'))
    Model.add(Dense(10, activation='elu'))
    Model.add(Dense(1))
    return Model


path = "Data"
data = TrainingFunctions.ImportData(path)
ImgPath , Steering = TrainingFunctions.ConvertDataPath(path,data)

Train_Img , Test_Img , Train_Steering , Test_Steering = train_test_split(ImgPath,Steering,test_size=0.2)
Train_X,Test_X = ConverPathtoImgArray(Train_Img,Test_Img)


CarModel = LoadModel()
CarModel.compile(optimizer='adam' , loss='mse' )
CarModel.fit(np.array(Train_X),Train_Steering,epochs=80 , batch_size=16 , steps_per_epoch=100 )

CarModel.save('CarModel.h5')

