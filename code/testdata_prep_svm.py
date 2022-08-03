import sys
sys.path.append('pymodules')
import pandas as pd
# this class read the raw input and tokenizes comprehensively for use with modeling
import pymodules.read_and_tokenize as contacts_utils
from sklearn.preprocessing import StandardScaler


def process_test_data(df, require_bigrams=True):
    """
    process test data frame for input into SVM model already generated
    :param df: Inout data in a dataframe format
    :param require_bigrams: Do we require bigrams or not
    :return: test data vector to be input into model.fit() function
    """
    prep_comments, df = contacts_utils.process_data_frame(df)
    if require_bigrams:
        for i in range(len(prep_comments.tokens)):
            prep_comments.tokens[i] = prep_comments.tokens[i] + prep_comments.bigrams[i]

    test_index = 0
    print(f"testdata_prep_svm: Comments at index[{test_index}] after addition of bigrams:\n {prep_comments.tokens[test_index]}")
    test_index = -1
    print(f"testdata_prep_svm: Comments at index[{test_index}] after addition of bigrams:\n {prep_comments.tokens[test_index]}")

    # document term matrix using count vectorizer
    dt_matrix, count_vectorizer = contacts_utils.create_document_term_matrix(prep_comments.tokens)
    dt_matrix = count_vectorizer.fit_transform(prep_comments.tokens)
    print(f"Document-term matrix created with shape: {dt_matrix.shape}")
    # we can access a dictionary that maps between words and positions of the document-term matrix.
    # We need this for SVM in order to make the word itself as column of dataframe (see output)
    id_word_indexer = pd.DataFrame(count_vectorizer.vocabulary_.items())

    ## FOR SVM, we need to make a matrix with proper column names
    # Also, we need another column that denotes the review
    # We also need to normalize the data
    df_svm = pd.DataFrame(dt_matrix.toarray())
    df_svm.rename(columns=id_word_indexer.to_dict()[0], inplace=True)
    ss = StandardScaler()
    df_svm_norm = ss.fit_transform(df_svm)
    return df_svm_norm

