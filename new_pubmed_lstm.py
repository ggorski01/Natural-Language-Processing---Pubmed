
#GPU reduce usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
#End of usage reduce code

#Lets create our csv files
def txtTocsv(inputFileName,outputFileName):
    #Open a csv to write
    csv = open(outputFileName,'w')
    #Write csv file heading
    csv.write("label,rawdata\n")
    #Open a text file to read
    f = open(inputFileName,'r')
    #For every line in file F
    for line in f:
        #if first line is not ###2222.... or an empty line, do:
        if '###' not in line and line!='\n':
            #replace commas by empty spaces
            line = line.replace(',',' ')
            #replace tab characters to comma so csv is well formatted
            line = line.replace('\t',',')
            #Append  modified line to the csv file
            csv.write(line)
    #Close file F(read)
    f.close()
    #Close file csv(write)
    csv.close()

#Calling the text to csv function
txtTocsv("test.txt","test.csv")
txtTocsv("train.txt","train.csv")
#Now we got our csv files

#From keras/imdb_lstm.py with pubmed
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import *
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from sklearn.metrics import confusion_matrix
from numpy import array2string
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

WEIGHTS_FINAL = 'model-pubmed-lstm-final.h5'
max_features = 10000  #cuts text after this number of words
maxlen = 30
batch_size = 32
MATRIX_NAME = 'confusion4.txt'
NUM_EPOCHS = 4

print("Loading pubmed data")

tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\]\^_`{|}~', lower=True)
#It cleans the dataset by removing special characters
#along with converting all words to lowercase.

#Working on training
tokenizer.fit_on_texts(train_df['rawdata'].values)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tokenizer.texts_to_sequences(train_df['rawdata'].values)
x_train = sequence.pad_sequences(x_train,maxlen = maxlen)
print('x_train shape:', x_train.shape)
print('x_train shape:', x_train.shape)
y_train = pd.get_dummies(train_df['label']).values
print('Shape of label tensor:', y_train.shape)

#Working on testing
x_test = tokenizer.texts_to_sequences(test_df['rawdata'].values)
x_test = sequence.pad_sequences(x_test,maxlen = maxlen)
print('x_test shape:', x_test.shape)
print('x_test shape:', x_test.shape)
y_test = pd.get_dummies(test_df['label']).values
print('Shape of label tensor:', y_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Print layers for resulting model
model.summary()

#Log training data into csv file
csv_logger = CSVLogger(filename="log.csv")
cblist = [csv_logger]

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,callbacks=cblist,
          epochs=NUM_EPOCHS,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)



predictions = model.predict(x_test) #the Validation set should work for now
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1)) #argmax uses the index of the maximum

confusionMatrix = open(MATRIX_NAME, 'w')
confusionMatrix.write(array2string(matrix))

# save trained model and weights
model.save(WEIGHTS_FINAL)