# Numeric Python
import numpy as np

# Natural Language ToolKit
import nltk

# System Libraries
import csv
import itertools
import os
import sys
import time
from datetime import datetime

# Global Variables
VOCABULARY_SIZE = 8000
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START_TOKEN"
SENTENCE_END_TOKEN = "SENTENCE_END_TOKEN"


# Reads data and appends SENTENCE_START AND SENTENCE_END Tokens
def read_data(filename="../data/reddit-comments-2015-08.csv", delimiter=","):
    print 'Reading CSV file..',filename
    fp = open(filename,'rb')
    data = csv.reader(fp, delimiter=delimiter, skipinitialspace=True)
    data.next()
    # Splits the comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(row[0].decode('utf-8').lower()) for row in data])
    # Add SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" %(SENTENCE_START_TOKEN, row, SENTENCE_END_TOKEN) for row in sentences]
    
    print "Parsed",len(sentences),"sentences"
    return sentences


def load_data(filename="../data/reddit-comments-2015-08.csv"):
    sentences = read_data()
    # Tokenize the sentences into words
    tok_sentences = np.array([nltk.word_tokenize(sentence) for sentence in sentences])

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tok_sentences))
    print 'Found',len(word_freq.items())," unique word tokens"

    # Getting the most common words and build index_to_word and word_to_index vectors
    vocabulary = np.array(word_freq.most_common(VOCABULARY_SIZE - 1))
    index_to_word = [row[0] for row in vocabulary]
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(word, index) for(index,word) in enumerate(index_to_word)])

    print 'Using Vocabulary of size',VOCABULARY_SIZE
    print 'The least frequent word in the vocabulary is "{0}"'.format(vocabulary[-1][0]) +      ' and appeared',vocabulary[-1][1],'times'

    # We replace all the words not in the vocabulary with "UNKNOWN_TOKEN"
    for index,sentence in enumerate(tok_sentences):
        tok_sentences[index] = [word if word in word_to_index else UNKNOWN_TOKEN for word in sentence]

    # print 'Example Sentence:',sentences[0]
    # print '\nSentence after pre-processing:', tok_sentences[0]

    # Creating the train data
    train_x = np.asarray([[word_to_index[word] for word in sentence[:-1]] for sentence in tok_sentences])
    train_y = np.asarray([[word_to_index[word] for word in sentence[1:]] for sentence in tok_sentences])

    return train_x, train_y