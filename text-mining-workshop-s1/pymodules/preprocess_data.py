import codecs
from os import path

with codecs.open(path.join(path.dirname(__file__), '../notebooks/auxiliary/stopwords_long.txt'),
                 'r', 'utf-8') as f:
    stp_long = set(f.read().splitlines())
with codecs.open(path.join(path.dirname(__file__), '../notebooks/auxiliary/stopwords_short.txt'),
                 'r', 'utf-8') as f:
    stp_short = set(f.read().splitlines())
