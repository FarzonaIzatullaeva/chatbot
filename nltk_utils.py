import numpy as np
# Algorith to reduce words to their base/root from (stem) by remiving common morphological suffixes. ex: "eating" becomes "eat"
from nltk.stem.porter import PorterStemmer
import nltk
# sentence tokinizer. Divides text into a list of individual sentences
nltk.download("punkt")


stemmer = PorterStemmer()


# splits continious text into individual words and punctuation marks which are then treated as "tokens"
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# get the words to their root form by stemming and converitng the words to lowercase


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    # take the tokenized sentences and stem each word.
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # create an array of zeros the same size as the array of all_words and make the digit types floats
    bag = np.zeros(len(all_words), dtype=np.float32)

    # gives a tuple list of words with corresponding index.
    for idx, w in enumerate(all_words):
        # if the word is present in the tokenized_sentence, change the index for that word in the touple from 0.0 to 1.0
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
