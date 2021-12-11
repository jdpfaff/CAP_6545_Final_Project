import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 

from fasta_reader import readFile
from helpers import *
from plot import *

# Variable
maxlen = 100
batch_size = 40
epochs = 400
seq_rows, seq_cols = 20, maxlen
num_classes = 2

# build CNN model
model = Sequential()
model.add(Conv2D(50, kernel_size=(10,6),activation='relu'))
model.add(Conv2D(25, kernel_size=(10,12),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(525, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=tensorflow.keras.optimizers.Adadelta(learning_rate=0.0075),metrics=['acc'])

print('Loading training data...')
pos_Train=readFile("./data/pos_training_dataset.txt",maxlen)
neg_Train=readFile("./data/neg_training_dataset_1.txt",maxlen)

print('Generating labels and features...')
(y_train, x_train) = createTrainTestData(pos_Train,neg_Train,"OneHot")

print('Shuffling the data...')
index=np.arange(len(y_train))
np.random.shuffle(index)
x_train=x_train[index,:]
y_train=y_train[index]

x_train = x_train.reshape(x_train.shape[0],seq_rows, seq_cols,1)
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)

x_train = tensorflow.cast(x_train,'float32')
print('Training...')

history = LossHistory()
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.1,callbacks=[history])

print('Plotting the acc-loss curve...')
history.loss_plot('epoch')

print('Loading test data...')
pos_Test = readFile("./data/pos_independent_test_dataset.txt",maxlen)
neg_Test = readFile("./data/neg_independent_test_dataset.txt",maxlen)

print('Generating labels and features...')
(y_test, x_test)=createTrainTestData(pos_Test,neg_Test,"Onehot")
x_test = x_test.reshape(x_test.shape[0],seq_rows,seq_cols,1)



print('Evaluating the model')
predicted_Probability = model.predict(x_test)
prediction = np.argmax(predicted_Probability, axis = 1)

print('Showing the confusion matrix')
cm=confusion_matrix(y_test,prediction)
print(cm)
print("ACC: %f "%accuracy_score(y_test,prediction))
print("F1: %f "%f1_score(y_test,prediction))
print("Recall: %f "%recall_score(y_test,prediction))
print("Pre: %f "%precision_score(y_test,prediction))
print("MCC: %f "%matthews_corrcoef(y_test,prediction))
print("AUC: %f "%roc_auc_score(y_test,prediction))
print('Plotting the ROC curve...')
plotROC(y_test,predicted_Probability[:,1])
results1 = [accuracy_score(y_test,prediction), f1_score(y_test,prediction), recall_score(y_test,prediction),
          precision_score(y_test,prediction), matthews_corrcoef(y_test,prediction), roc_auc_score(y_test,prediction)]
print('results')
print(results1)
print('saving the model...')
model.save_weights('./models_weights/model_weights.h5')
with open('./models_weights/json_model.json', 'w') as f:
     f.write(model.to_json())
