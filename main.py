from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

X = np.array([],dtype=np.int32)
y = np.array([],dtype=np.int32)
model = None

def record_object(X, y, images=100):
    cap = cv2.VideoCapture(0)
    for i in range(images):
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X = np.append(X, image)
        y = np.append(y, 1)
    X = X.reshape(images,640,480,3)
    print(X)
    cap.release()
    os.system("cls")
    print("OBJECT RECORDED SUCCESFULLY!")
    return X, y

def record_other(X, y, images=100):
    already_images = len(X)
    cap = cv2.VideoCapture(0)
    for i in range(images):
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X = np.append(X, image)
        y = np.append(y, 0)
    X = X.reshape(already_images+images,640,480,3)
    print(X)
    cap.release()
    os.system("cls")
    print("OTHER OBJECT RECORDED SUCCESFULLY")
    return X, y

def train(X,y):
    print("MODEL TRAINING... (this will take a while)")
    y = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Flatten the feature maps
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=X_train,y=y_train, epochs=5)
    os.system("cls")
    print("MODEL TRAINED!")
    return model

def test():
    print("Hold one of the objects in front of the camera.")
    cap = cv2.VideoCapture(0)
    numbers = []
    for i in range(10):
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(1,640,480,3)
        numbers.append(model.predict(image).argmax())
    cap.release()
    Result = 0
    for i in numbers:
        Result += i

    Result = Result / len(numbers)
    os.system("cls")
    if Result >= 0.5:
        print(f"The CNN detected this is {object}")
    else:
        print(f"The CNN detected this is {object2}")
    input()


while True:
    main = input("Press 1 to record an object, 2 to record another object, and 3 to train the model, 4 to test the model >> ")
    if main == "1":
        object = input("What object is this?")
        X, y = record_object(X, y)
    elif main == "2":
        object2 = input("What object is this?")
        X , y = record_other(X, y)
    elif main == "3":
        if len(y) > 0:
            model = train(X,y)
        else:
            print("You need to record some data first! (two objects)")
    elif main == "4":
        if model != None:
            test()
        else:
            print("Model not trained yet!")
    else:
        print("Incorrect input")



