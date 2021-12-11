from keras.models import model_from_json
from fasta_reader import readFile
from helpers import *
import numpy as np

maxlen = 100
seq_rows, seq_cols = 20, maxlen


print('Loading model...')

with open('./models_weights/json_model.json', 'r') as f:
     json_string = f.read()

DeepT3_1 = model_from_json(json_string)
DeepT3_1.load_weights('./models_weights/model_weights.h5')

with open('./models_weights/json_model_2.json', 'r') as f:
     json_string = f.read()

DeepT3_2 = model_from_json(json_string)
DeepT3_2.load_weights('./models_weights/model_weights_2.h5')

print('Loading data...')
testData = readFile("./data/P.syringae_nr_effector.txt",maxlen)

print('Generating features...')
x_test = createData(testData,"Onehot")
x_test = x_test.reshape(x_test.shape[0],seq_rows, seq_cols,1)


print('Predicting...')
predicted_Probability_1 = DeepT3_1.predict(x_test)
predicted_Probability_2 = DeepT3_2.predict(x_test)
prediction_1 = np.argmax(predicted_Probability_1, axis = 1)
prediction_2 = np.argmax(predicted_Probability_2, axis = 1)
prediction = prediction_1 + prediction_2 

print('Saving the result..')
f = open ("result_p.txt", "w")
for i in prediction:
    if i == 2:
      f.write("T3SE\n")
        
    else:
      f.write("non-T3SE\n")

f.close()
print('done')









