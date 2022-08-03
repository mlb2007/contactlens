import pandas as pd
import string
import sys
import re
sys.path.append('pymodules')
# This class contains some utility functions Word2Vec, stop words etc. etc.
import preprocessing_class as pc
# gender gueser
import gender_guesser.detector as gd
# for dictionary method synonym finder using wordnet
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import multilabel_confusion_matrix, f1_score
# multiprocessing
import multiprocess as mp
from sklearn.feature_extraction.text import CountVectorizer
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def get_token_weights(tokens, gensim_model, num_expected_unique_words=10000, MAX_SEQ_LEN=300):
    """
    Prepare RNN inout given test tokens
    :param tokens: Processed data into token
    :param gensim_model: this is the word embedding model that must have been created when we train on inputs
    :param num_expected_unique_words: input obtained from how we trained the RNN model (default = 10000)
    :param MAX_SEQ_LEN: input obtained from how we trained the model (default = 300)
    :return: padded ouput that can be used in model.fit() and the gensim weight matrix required for RNN model
    """
    # because embedding is independent of tokenization, we integerize our token based on keras tokenizer
    keras_tokenizer = Tokenizer(num_expected_unique_words, split=",")
    keras_tokenizer.fit_on_texts(tokens)
    X = tokens
    X_train=keras_tokenizer.texts_to_sequences(X) # this converts texts into some numeric sequences
    X_train_pad=pad_sequences(X_train,maxlen=MAX_SEQ_LEN,padding='post') # this makes the length of all numeric sequences equal

    # extract the word embeddings from the model
    word_vectors = gensim_model.wv
    word_vectors_weights = gensim_model.wv.vectors
    vocab_size, embedding_size = word_vectors_weights.shape
    gensim_weight_matrix = np.zeros((num_expected_unique_words ,embedding_size))
    for word, index in keras_tokenizer.word_index.items():
        if index < num_expected_unique_words: # why ? since index starts with zero
            try:
                word_index_in_embedding = word_vectors.key_to_index[word]
            except KeyError:
                gensim_weight_matrix[index] = np.zeros(embedding_size)
            else:
                gensim_weight_matrix[index] = word_vectors[word_index_in_embedding]

    return X_train_pad, gensim_weight_matrix


def create_document_term_matrix(tokens):
    # count vectorizer
    # simple auxiliary function to override the preprocessing done by sklearn
    def do_nothing(doc):
        return doc

    # create a CountVectorizer object using our preprocessed text
    # uni gram
    count_vectorizer = CountVectorizer(encoding='utf-8',
                                       preprocessor=do_nothing,  # apply no additional preprocessing
                                       tokenizer=do_nothing,  # apply no additional tokenization
                                       lowercase=False,
                                       strip_accents=None,
                                       stop_words=None,
                                       ngram_range=(1, 1),  # generate only unigrams
                                       analyzer='word',  # analysis at the word-level
                                       # max_df=0.5,              # ignore tokens that have a higher document frequency (can be int or percent)
                                       # min_df=500,              # ignore tokens that have a lowe document frequency (can be int or percent)
                                       min_df=10,
                                       max_features=None,  # we could impose a maximum number of vocabulary terms
                                       )
    # transform our preprocessed tokens into a document-term matrix
    dt_matrix = count_vectorizer.fit_transform(tokens)
    return dt_matrix, count_vectorizer


def find_wordnet_synonyms(word_list, type_of_word=None):
    """
    Find synonyms of words given as a list in the input word_list.
    Sometimes, we want synonyms with matching type so that some words that have meaning as noun and adjective,
    for example, are correctly filtered
    It is assumed that the word_list words are themselves synonyms of each other and is given as a list
    return lemmatized synonyms ...
    """
    synonyms = set()
    for word_to_look in word_list:
        #print(f"looking for synonyms of word:{word_to_look}")
        for syn in wn.synsets(word_to_look, pos=type_of_word):
            for i in syn.lemmas():
                synonyms.add(i.name())
    #print(f"Synonyms:\n {synonyms}")
    return synonyms

def first_name(x):
    """
    Function to get the first name so that we can guess the gender
    We determine the first name from the given string. We also remove any digits from the name.
    Further, we use space to split names
    """
    x_split = str(x).split()
    fname = x_split[0]
    # remove reference to digits. Now after removal, there could be some misclassification, but that is ok ..
    fname_p = re.sub(r'[0-9]+', "", fname)
    ret_str = fname_p.capitalize()
    return ret_str


def read_file(filename):
    """
    Read Excel sheet, tokenize and return object with tokens ...
    """
    print(f"Read sheet 'Scrubbed_data' ...")
    df_raw = pd.read_excel(filename, sheet_name='Scrubbed_data', index_col='REVIEW_DATE')
    return process_data_frame(df_raw)


def process_data_frame(df):
    """
    :param df: Input is any dataframe that contains user data
    :return: processed tokens and the dataframe processed
    """
    not_needed_columns = ['OVERALL_RATING', 'COMFORT_RATING', 'VISION_RATING', 'VALUE_FOR_MONEY', 'PROS', 'CONS',
                  'ORIGINAL_SOURCE', 'REPLY_FROM_ACCUVUE',
                  'PRODUCT_LINK', 'WEBSITE']
    print(f"Columns dropped: {not_needed_columns}")
    df = df.drop(columns = not_needed_columns, axis=1)

    """
    Let us figure out the gender from the names and drop the names column
    # We use gender_guesser package.
    """
    gdx = gd.Detector()
    df['GENDER'] = df.AUTHOR.apply(first_name).map(lambda x: gdx.get_gender(x))
    print(f" Drop the Author column and replace it with gender of author ...")
    df.drop(columns=['AUTHOR'], axis=1, inplace=True)

    print("Consolidate all the comments into one column called COMMENT")
    # Comments can occur both in title and in Comment columns.
    df['COMMENT'] = df['TITLE'].astype(str).fillna("") + " " + df['COMMENTS'].astype(str).fillna("")
    df.drop(columns = ['TITLE', 'COMMENTS'], axis=1, inplace=True)

    print("Make ratings into integers")
    # replace N = No rating with 0. We do this because rating is assumed to be numeric, not categorical
    df['RATING'].replace('N', '0', inplace=True)
    # convert rating to integers
    df['RATING'] = df['RATING'].apply(lambda x: int(x))

    ## regex for tokenization
    # Ref: http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py
    emoticon_string = r"""
        (?:
          [<>]?
          [:;=8]                     # eyes
          [\-o\*\']?                 # optional nose
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          |
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          [\-o\*\']?                 # optional nose
          [:;=8]                     # eyes
          [<>]?
        )"""

    # The components of the tokenizer:
    regex_strings = (
        # Phone numbers:
        r"""
        (?:
          (?:            # (international)
            \+?[01]
            [\-\s.]*
          )?
          (?:            # (area code)
            [\(]?
            \d{3}
            [\-\s.\)]*
          )?
          \d{3}          # exchange
          [\-\s.]*
          \d{4}          # base
        )"""
        ,
        # Emoticons:
        emoticon_string
        ,
        # HTML tags:
        r"""<[^>]+>"""
        ,
        # Twitter username:
        r"""(?:@[\w_]+)"""
        ,
        # Twitter hashtags:
        r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
        ,
        # Remaining word types:
        r"""
        (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
        |
        (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
        |
        (?:[\w_]+)                     # Words without apostrophes or dashes.
        |
        (?:\.(?:\s*\.){1,})            # Ellipsis dots.
        |
        (?:\S)                         # Everything else that isn't whitespace.
        """,
        r"""
        (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
        \w+(?:-\w+)*        # preserve expressions with internal hyphens as single tokens
        | [][.,;"'?():-_`]  # preserve punctuation as separate tokens
        """
    )
    word_re = re.compile(pattern=r"""(%s)""" % "|".join(regex_strings), flags=re.VERBOSE | re.I)

    print("Tokenize data based on regex found from experimentation and common usage ...")
    comments_data = df.COMMENT
    prep_comments = pc.RawDocs(comments_data,  # series of documents
                      lower_case=True,  # whether to lowercase the text in the firs cleaning step
                      stopwords='long',  # type of stopwords to initialize
                      contraction_split=True,  # wheter to split contractions or not
                      tokenization_pattern=word_re  # custom tokenization patter
                      )

    ## Test index to check if what we are doing makes sense or not ..
    test_index = 0
    #comments_data
    print(f"Comments before tokenization at index[{test_index}]:\n {comments_data[test_index]}")
    print(f"Comments after tokenization at index[{test_index}]:\n {prep_comments.docs[test_index]}")

    # lower-case text, expand contractions and initialize stopwords list
    stop_lens_list = ['Lens', 'lens', 'Contact-lens', 'Contact-Lens', 'lenses', 'Lenses', 'Contact-Lenses',
                      'Contact-lenses', 'contact-lens', 'contact-lenses', 'acucue', 'acuvue', 'Acuvue', 'pack',
                      'box', 'Pack', 'Box', 'Moist', 'moist', 'month', 'trial', 'lens.com', 'Lens.com']
    prep_comments.basic_cleaning(custom_stopwords_list=stop_lens_list)
    #prep_comments.basic_cleaning()

    # test after explore an example after the basic cleaning has been applied
    print(f"Comments at index[{test_index}] before basic cleaning:\n {comments_data[test_index]}")
    print(f"Comments at index[{test_index}] after cleaning:\n {prep_comments.docs[test_index]}")

    # now we can split the documents into tokens
    prep_comments.tokenize_text()

    # test tokens ...
    i = 0
    print(comments_data[i])
    print()
    print(prep_comments.tokens[i])

    # Replace punctuation ...
    punctuation = string.punctuation
    punctuation = punctuation.replace("-", "") # remove the hyphen from the punctuation string
    # punctuation
    # clean punctuation
    prep_comments.token_clean(length=2,                 # remove tokens with less than this number of characters
                     punctuation=punctuation,           # remove custom list of punctuation characters
                     numbers = True                     # remove numbers
                     )
    # test
    print(f"Comments at index[{test_index}] before removal of punctuations:\n {comments_data[test_index]}")
    print(f"Comments at index[{test_index}] after removal of punctuation:\n {prep_comments.tokens[test_index]}")

    # we need to specify that we want to remove the stopwords from the "tokens"
    # tokens is the name of attribute that contains all the tokens prep_comments
    prep_comments.stopword_remove('tokens')

    # test
    print(f"Comments at index[{test_index}] before removal of stop words:\n {comments_data[test_index]}")
    print(f"Comments at index[{test_index}] after removal of stop words:\n {prep_comments.tokens[test_index]}")

    print("Lemmatize tokens")
    # stemming
    # pre_comments.stem()
    # apply lemmatization to all documents (takes a very long time so we will avoid it for now)
    prep_comments.lemmatize()
    print(f"Comments at index[{test_index}] after lemmatization:\n {prep_comments.lemmas[test_index]}")

    print("Build the bigram and trigram words for use with topic modeling")
    prep_comments.bigram('tokens')
    prep_comments.trigram('tokens')

    # not sure why this is needed as it does not seem to be used anyway, nevertheless ...
    df['FINAL_PRODUCT_NAME'] = df.FINAL_PRODUCT_NAME.values[0].strip(" \t")

    return prep_comments, df

### utility functions
# for parallel run of multiple trials ...
def run_parallel(func, num_cpus=4):
    """
    A simple parallel processor
    """
    mp_pool = mp.Pool(num_cpus)

    def _run(grid_parameters):
        result = mp_pool.map(func, grid_parameters)
        return result

    return _run


# Ref: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/
def onehot_encode(data):
    """
    one hot encoding for CNN
    """
    return to_categorical(data)


# load dataset
def split_data(X, y, validation=False, shuffle=False):
    """
    load data and create validation set as well (25% of training data)
    """
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=shuffle)
    # create validation data as well
    if validation:
        trainX, validX, trainy, validy = train_test_split(trainX, trainy, test_size=0.25, random_state=42,
                                                          shuffle=shuffle)
        return trainX, trainy, testX, testy, validX, validy
    else:
        return trainX, trainy, testX, testy, None, None


def plot_accuracy(model, test_str='Validation'):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', test_str], loc='upper left')
    plt.show()


def plot_loss(model, test_str='Validation'):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', test_str], loc='upper left')
    plt.show()


def convert_prob_to_labels(data):
    # the outputs are probabilities, change it to classifier labels ..
    y_arg = np.argmax(data, axis=1)
    y_pred = onehot_encode(y_arg)
    return y_pred


def model_stats_all_labels(Y_pred, Y_actual):
    """
    Returned confusion matrices will be in the order of sorted unique labels in the union of(y_true, y_pred)
    """
    # true negatives is  C[0, 0]
    # false negatives is C[1, 0]
    # true positives is  C[1, 1]
    # false positives is C[0, 1]
    mcm = multilabel_confusion_matrix(Y_actual, Y_pred)
    tnv = mcm[:, 0, 0]
    tpv = mcm[:, 1, 1]
    fnv = mcm[:, 1, 0]
    fpv = mcm[:, 0, 1]

    accuracy = (tpv + tnv) / (tpv + tnv + fpv + fnv)
    sensitivity = tpv / (tpv + fnv)
    specificity = tnv / (tnv + fpv)
    denom = 1 - specificity
    likelihood = [sensitivity[i] / denom[i] if denom[i] > 0 else np.nan for i in range(len(denom))]

    return accuracy, sensitivity, specificity, likelihood

