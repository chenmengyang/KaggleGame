from main import load_data
from models.rnn import lstm_model

trainX, trainY, test, emb, maxlen = load_data()
print ('trainX shape is {}'.format(trainX.shape))
print ('trainY shape is {}'.format(trainY.shape))
print ('test shape is {}'.format(test.shape))
print ('maxlen is {}'.format(maxlen))
model = lstm_model((maxlen,), emb[1], emb[0])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs = 50, batch_size = 10000, shuffle=True)