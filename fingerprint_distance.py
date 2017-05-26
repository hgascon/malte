# MALTE
# Evaluation functions for experiment results
# (c) 2017 Hugo Gascon

from simhash import Simhash


def ngrams(tokens):
    """
    Find all ngrams within a series of tokens

    :param tokens: a list of strings
    """
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+3, min(n_tokens, i+3)+1):
            yield tokens[i:j]


def get_features(facts):
    return list(ngrams(' '.join(facts)))


def get_hash(facts):
    return Simhash(get_features(facts))


def distance(f1, f2):
    return get_hash(f1).distance(get_hash(f2))
