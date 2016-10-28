# coding=utf-8

import json
import re
import os
import numpy as np
from toolz import itemmap

from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD


PADDING = '<pad>'
UNKNOWN = 'UNK'
EOA = '<eos>'       # end of answer
EOQ = '<eoq>'       # end of question
EXTRA_WORDS_NAMES = [PADDING, UNKNOWN, EOA, EOQ]
EXTRA_WORDS = {PADDING:0, UNKNOWN:1, EOA:2, EOQ:3}
EXTRA_WORDS_ID = itemmap(reversed, EXTRA_WORDS)
MAXLEN = 50

OPTIMIZERS = { \
        'sgd':SGD,
        'adagrad':Adagrad,
        'adadelta':Adadelta,
        'rmsprop':RMSprop,
        'adam':Adam,
        }
###
# Functions
###
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=len(EXTRA_WORDS))
def _myinc(d):
    """
    Gets a tuple d, and returns d[0]: id.
    """
    x = d[0]
    _myinc.counter += 1
    return (x, _myinc.counter - 1)


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory

def preprocess_line(line):
    cap_tmp = line.strip().decode('utf-8').lower().encode('utf8')
    return cap_tmp

def preprocess_caption(cap):

    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    cap_tmp = cap.strip().decode('utf-8').lower().encode('utf8')
    cap_tmp = processPunctuation(cap_tmp)
    return cap_tmp


def preprocess_question(q):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                    "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t",
                    "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve",
                    "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve",
                    "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve",
                    "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve",
                    "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t",
                    "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", "mustnt": "mustn’t",
                    "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t",
                    "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t",
                    "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve",
                    "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve",
                    "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve",
                    "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d",
                    "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll",
                    "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve",
                    "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s",
                    "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve", "therere": "there’re",
                    "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve",
                    "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t",
                    "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll",
                    "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered":
                        "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d", "whod’ve": "who’d’ve",
                    "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll",
                    "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t",
                    "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll",
                    "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve",
                    "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll",
                    "youre": "you’re", "youve": "you’ve"}
    manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
                 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    articles = ['a', 'an', 'the']
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText

    q_tmp = q.strip().lower().encode('utf8')
    #q_tmp = processPunctuation(q_tmp)
    #q_tmp = processDigitArticle(q_tmp)
    if q_tmp[-1] == '?' and q_tmp[-2] != ' ':
        # separate word token from the question mark
        q_tmp = q_tmp[:-1] + ' ?'
    # remove question mark
    if q_tmp[-1] == '?': q_tmp = q_tmp[:-1]
    return q_tmp


def save_txt_answers(samples, savefile='./sample', whichset='val', step=''):
        with open(savefile + '_' + whichset + '_samples_' + str(step) + '.json', 'w') as f:
            print >>f, '\n'.join(samples)


def save_json_answers(samples, savefile='./sample', whichset='val', step=''):
        with open(savefile + '_' + whichset + '_samples_' + str(step) + '.json', 'w') as f:
            json.dump(samples, f)
def build_vocabulary(this_wordcount, extra_words=EXTRA_WORDS,
        is_reset=True, truncate_to_most_frequent=0):
    """
    Builds vocabulary from wordcount.
    It also adds extra words to the vocabulary.

    In:
        this_wordcount - dictionary of wordcounts, e.g. {'cpu':3}
        extra_words - additional words to build the vocabulary
            dictionary of {word: id}
            by default {UNKNOWN: 0}
        is_reset - if True we restart the vocabulary counting
            by defaults False
        truncate_to_most_frequent - if positive then the vocabulary
            is truncated to 'truncate_to_most_frequent' words;
            by default 0
    Out:
        word2index - mapping from words to indices
        index2word - mapping from indices to words
    """
    if is_reset:
        _myinc.counter=len(EXTRA_WORDS)
    if truncate_to_most_frequent > 0:
        sorted_wordcount = dict(sorted(
                this_wordcount.items(), key=lambda x:x[1], reverse=True)[:truncate_to_most_frequent])
        this_wordcount = sorted_wordcount
    word2index = itemmap(_myinc, this_wordcount)
    if not extra_words == {}:
        assert(all([el not in word2index.values() for el in extra_words.values()]))
        word2index.update(extra_words)
    index2word = itemmap(reversed, word2index)
    return word2index, index2word


def index_sequence(x, word2index):
    """
    Converts list of words into a list of its indices wrt. word2index, that is into
    index encoded sequence.

    In:
        x - list of lines
        word2index - mapping from words to indices

    Out:
        a list of the list of indices that encode the words
    """
    one_hot_x = []
    for line in x:
        line_list = []
        for w in line.split():
            w = w.strip()
            if w in word2index: this_ind = word2index[w]
            else: this_ind = word2index[UNKNOWN]
            line_list.append(this_ind)
        one_hot_x.append(line_list)
    return one_hot_x

