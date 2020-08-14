#%% Imports
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import pandas as pd
from datetime import datetime

#%% Helper Functions

def norm(array):
   return array/9

def invNorm(array):
   return 9*array

#%% Processing Data

games = pd.read_csv("./data/sudoku.csv")

quizzes = games['quizzes']
solutions = games['solutions']

features = []
labels = []

fraction_used = 1

for i in range(int(len(quizzes)/fraction_used)):
   temp_arr = [norm(int(char)) for char in quizzes[i]]
   features.append(np.array(temp_arr).reshape((9,9,1)))

   if(i%200000==0):
       print(str(i/20000) + "%")



#We represent a 1 as label 0, 2 as label 1, etc.

for i in range(int(len(solutions)/fraction_used)):
   temp_arr = [(int(char)-1) for char in solutions[i]]
   labels.append(np.array(temp_arr).reshape((81,1)))
   if(i%200000==0):
       print(str(i/20000 +50)+ "%")


features = np.array(features)
labels = np.array(labels)
print("100%")
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

#%% Creating Model

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size=(1,1), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(81*9))
    model.add(layers.Reshape((81, 9)))
    model.add(layers.Activation('softmax'))
    return model

# model = create_model()
# model.summary()

#%% Loading Model

model = create_model()
model.load_weights('./checkpoints/latest_checkpoint')
model.summary()

#%%Logging Loss

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#%% Model Training

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

#Tensorboard loading
os.system('cmd /k "tensorboard --logdir logs/scalars"')

#%% Saving Model

model.save_weights('checkpoints/latest_checkpoint')

#%% Defining the solver functions

def step_by_step(inputGame):

    finished = False
    while not finished:

        predictions = model.predict(inputGame.reshape((1,9,9,1)))[0]
        #Unnormalize the predictions and square values
        predict = np.argmax(predictions, axis=1).reshape((9,9))+1
        prob = np.max(predictions, axis=1).reshape((9,9))
        inputGame = invNorm(inputGame).reshape((9,9))
        missing = (inputGame==0)

        if missing.sum()==0:
            finished = True
        else:
            max = 0
            ind = 0
            for i in range(9):
                for j in range(9):
                    if missing[i][j] and prob[i][j]>= max:
                        max = prob[i][j]
                        ind = i*9 + j
                    elif missing[i][j] and prob[i][j]>0.8:
                        inputGame[i][j] = predict[i][j]
            inputGame[int(ind/9)][(ind%9)] = predict[int(ind/9)][(ind%9)]
            inputGame = norm(inputGame)

    return predict



def solve(game):
    game = norm(np.array([int(j) for j in game]).reshape((9,9,1)))
    game = step_by_step(game)
    return game

#%% Testing

test_board = '004300209005009001070060043006002087190007400050083000600000105003508690042910300'

solved = solve(test_board)

print(solved)

#%% End

print("End")


