#-*- coding: utf-8 -*-

import numpy as np
import utils
from spanishGenWin import spanishGenWin as swg #Script for creating Gender Windows
from spanishNumWin import spanishNumWin as swn #Script for creating Number Windows 
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.optimizers import Adagrad, SGD
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm

#Path were the corpus is located
PATH = 'corpus/' 

#Size of the vocabulary to use
vocabSize = 5000

#Windows size
winSize = 5

VSIZE = 100
nb_epoch = 15
batch_size = 128
print "Processing corpus"

#Change the function accordingly to the task (swn, swg)
training = swn(PATH + 'train/train.gennum.es', \
    PATH + 'train/train.es', \
    vocabSize,winSize)

dev = swn(PATH + 'dev/dev.gennum.es', \
    PATH + 'dev/dev.es', \
	vocabSize,winSize)

test = swn(PATH + 'test/test.gennum.es', \
    PATH + 'test/test.es', \
	vocabSize,winSize)

trainWindows, trainTargets = training.process()
devWindows, devTargets  = dev.process()
testWindows, testTargets  = test.process()

vocabulary = utils.getVocabulary(trainWindows,winSize,vocabSize)

trainFeatures = utils.vectorizeWindows(trainWindows,vocabulary)
devFeatures = utils.vectorizeWindows(devWindows,vocabulary)
testFeatures = utils.vectorizeWindows(testWindows,vocabulary)

trainTargets = np.asarray(trainTargets)
devTargets = np.asarray(devTargets)
testTargets = np.asarray(testTargets)

print "Finished processing"

model = Sequential()
# Number of embedding vectors = vocabSize + UNK + <s> + <e>
model.add(Embedding(vocabSize + 3, VSIZE, input_length=winSize, input_dtype='int32'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(trainTargets.shape[1], activation='softmax'))
model.summary()
adagrad=Adagrad(lr=0.0008, epsilon=1e-5, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])

tr_loss = []
val_loss = []
tr_acc = []
val_acc = []
# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(trainFeatures, trainTargets,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=1, validation_data=(devFeatures, devTargets))
    tr_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    tr_acc.extend(history.history['acc'])
    val_acc.extend(history.history['val_acc'])
    print
    preds = model.predict_classes(devFeatures, verbose=0)
np.savetxt('tr_losses.txt', tr_loss)
np.savetxt('val_losses.txt', val_loss)
np.savetxt('tr_acc.txt', tr_acc)
np.savetxt('val_acc.txt', val_acc)

score = model.evaluate(testFeatures, testTargets, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
