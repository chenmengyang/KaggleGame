# Todo Lists

1. data pre-processing
* find the maximun length sentences from training set
* apply padding to all train + test dataset, so all the sentences will have the same length (because input layer can only accept same fixed input size)


2. find pretrain word-embedding (please read [here](https://www.tensorflow.org/tutorials/representation/word2vec) to understand why we need word-embedding)
* good word-embedding can present the meaning of words
* word-embedding will convert charater words into integers with meanings

3. design neural network architect
* better test with LSTM, RNN, GRU
* implement with Keras or tensorflow


# Notes

1. There are some errors with the data (both train and test), I think its because the embedding matrix (which has learned 400000 words) didn't learn some of the words, should check carefully, e.g. a word like 'pre-train' is hard to deal with, but we can absoluty convert it into 2 words 'pre' and 'train'.
* there are 13548 rows of error sentences in train data
* there are 5361 rows of error sentences in test set
