import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
from collections import Counter
import random
import ast
import re
import scipy

sys.path.append('pymodules')
# This class contains some utility functions Word2Vec, stop words etc. etc.
import pymodules.preprocessing_class as pc

# gender gueser
import gender_guesser.detector as gd

# for dictionary method synonym finder using wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

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

def read_file(filename, filter_columns):
    """
    Read Excel sheet and filter uneeded columns as shown
    """
    df_raw = pd.read_excel(filename, sheet_name='Scrubbed_data', index_col='REVIEW_DATE')
    df = df_raw.drop(columns = filter_columns, axis=1)
    return df

def detect_gender(df):
    """
    Let us figure out the gender from the names and drop the names column
    # We use gender_guesser package.
    """
    gdx = gd.Detector()
    df['GENDER'] = df.AUTHOR.apply(first_name).map(lambda x: gdx.get_gender(x))
    # Drop the author column now
    df.drop(columns=['AUTHOR'], axis=1, inplace=True)
    return df

def make_comments_column(df):
    # Consolidate the comments into one column
    # Comments can occur both in title and in Comment columns.
    df['COMMENT'] = df['TITLE'].astype(str).fillna("") + " " + df['COMMENTS'].astype(str).fillna("")
    df.drop(columns = ['TITLE', 'COMMENTS'], axis=1, inplace=True)
    return df

def make_rating_numerical(df):
    # clean rating
    # replace N = No rating with 0. We do this because rating is assumed to be numeric, not categorical
    df['RATING'].replace('N', '0', inplace=True)
    # convert rating to integers
    df['RATING'] = df['RATING'].apply(lambda x: int(x))
    return df

def make_regex():
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
    return word_re

def make_tokens(df, word_regex):
    comments_data = df.COMMENT
    prep_comments = pc.RawDocs(comments_data,  # series of documents
                      lower_case=True,  # whether to lowercase the text in the firs cleaning step
                      stopwords='long',  # type of stopwords to initialize
                      contraction_split=True,  # wheter to split contractions or not
                      tokenization_pattern=word_regex  # custom tokenization patter
                      )
    ## test ...
    #comments_data
    i = 0
    print("Document from the pandas series:\n", comments_data[i])
    print("\n-------------------------\n")
    print("Document from preprocessing object:\n", prep_comments.docs[i])

    # lower-case text, expand contractions and initialize stopwords list
    prep_comments.basic_cleaning()

    # test after explore an example after the basic cleaning has been applied
    i = 0
    print(comments_data[i])
    print()
    print(prep_comments.docs[i])

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

    i = 0
    print(comments_data[i])
    print()
    print(prep_comments.tokens[i])

    # get the list of stopwords provided earlier
    # print(sorted(prep_comments.stopwords))

    # we need to specificy that we want to remove the stopwords from the "tokens"
    # tokens is the name of attribute that contains all the tokens prep_comments
    prep_comments.stopword_remove('tokens')

    # test
    i = 0
    print(comments_data[i])
    print()
    print(prep_comments.tokens[i])

    # lemmatize ....
    # stemming
    # pre_comments.stem()
    # apply lemmatization to all documents (takes a very long time so we will avoid it for now)
    prep_comments.lemmatize()

    # Build the bigram and trigram models
    prep_comments.bigram('tokens')
    prep_comments.trigram('tokens')

    return prep_comments


