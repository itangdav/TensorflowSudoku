#%% Imports
import os

import progressbar as progressbar
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from datetime import datetime
import queue
import threading
import multiprocessing
import subprocess


#%% GPU Stuff
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#%% Helper Functions

def norm(array):
   return (array/9-0.5)

def invNorm(array):
   return 9*(array+0.5)

#%% Processing Data

games = pd.read_csv("./data/sudoku.csv")

quizzes = games['quizzes']
solutions = games['solutions']
quizzes, solutions = shuffle(np.array(quizzes), np.array(solutions))

features = []
labels = []

fraction_used = 100

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
    model.add(layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(81*9))
    model.add(layers.Reshape((81, 9)))
    model.add(layers.Activation('softmax'))
    return model

model = create_model()

#%% Loading Model
#
model.load_weights('./checkpoints/latest_checkpoint')



model.summary()

#%%Logging Loss

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#%% Model Training

model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])



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
                    elif missing[i][j] and prob[i][j]>=1:
                        print("once")
                        inputGame[i][j] = predict[i][j]
            inputGame[int(ind/9)][(ind%9)] = predict[int(ind/9)][(ind%9)]
            inputGame = norm(inputGame)

    return predict



def solve(game):
    game = norm(np.array([int(j) for j in game]).reshape((9,9,1)))
    game = step_by_step(game)
    return game

#%% Testing

total = 100 #must be <1 mil
global correct
correct = 0
quizzes, solutions = shuffle(np.array(quizzes), np.array(solutions))

#Multithreaded processing
num_of_cores = multiprocessing.cpu_count()

q = queue.Queue()
for i in range(total):
    q.put([quizzes[i], solutions[i]])

def worker():
    global correct
    while True:
        item = q.get()
        test_board = item[0]
        test_solution = item[1]

        test_solution = np.array([int(j) for j in test_solution]).reshape((9, 9))

        solved = solve(test_board)

        if (np.all(solved - test_solution == 0)):
            correct += 1

        q.task_done()


print("Creating %d threads" % num_of_cores)
for i in range(num_of_cores):
     t = threading.Thread(target=worker)
     t.daemon = True
     t.start()


q.join() #Check if all finished

print("Solved ", 100 * correct / total,"%")

#%% End

print("End")


#Tensorboard loading
os.system('cmd /k "tensorboard --logdir logs/scalars"')