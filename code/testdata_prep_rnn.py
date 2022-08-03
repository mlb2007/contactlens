import sys
sys.path.append('pymodules')
# this class read the raw input and tokenizes comprehensively for use with modeling
import pymodules.read_and_tokenize as contacts_utils


def process_test_data(df, gensim_model, require_bigrams=True, num_expected_unique_words = 10000, MAX_SEQ_LEN = 300):
    """
    Process test data given as df to input into model.fit() for RNN model
    :param df: Inout dataframe
    :param require_bigrams: Do we need to add bigrams as well ?
    :return: processed tokens, processed data frame
    """
    prep_comments, df = contacts_utils.process_data_frame(df)
    if require_bigrams:
        for i in range(len(prep_comments.tokens)):
            prep_comments.tokens[i] = prep_comments.tokens[i] + prep_comments.bigrams[i]

    test_index = 0
    print(f"Comments at index[{test_index}] after addition of bigrams:\n {prep_comments.tokens[test_index]}")
    test_index = -1
    print(f"Comments at index[{test_index}] after addition of bigrams:\n {prep_comments.tokens[test_index]}")

    X, weights = contacts_utils.get_token_weights(prep_comments.tokens, gensim_model, num_expected_unique_words, MAX_SEQ_LEN)
    return X, weights
