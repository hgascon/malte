# MALTE
# Evaluation functions for experiment results
# (c) 2017 Hugo Gascon

import random
from lxml.etree import XMLSyntaxError
from progressbar import Percentage, Bar, ETA, ProgressBar
from scipy.spatial.distance import squareform
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score
from lxml import etree

from average_precision import mapk
from collections import defaultdict
import numpy as np
import os
import pz


DATA_DIR = 'results'
MATRICES = ['factlist_matrix.pz', 'subgraph_matrix.pz', 'facts_matrix.pz']
LABELS = ['factlist_labels.pz', 'subgraph_labels.pz', 'facts_labels.pz']
RESULTS = ['results/map_factlist.pz',
           'results/map_subgraph.pz',
           'results/map_facts.pz']


def run_experiment():
    """ Wrapper call to map_range_evaluation to evaluate the average MAP
    in the dataset for all matrices and save the results in the
    corresponding files. In the case of 'facts' a random permutation
    is applied to the matrix and a sampling is done to reduce the
    computation time (30000 facts ~ 19 hours)
    """
    print "Running MAP evaluation.."
    for m, l, fname in zip(MATRICES, LABELS, RESULTS):
        data = os.path.join(DATA_DIR, m)
        labels = os.path.join(DATA_DIR, l)
        print "[*] Loading data matrix..."
        X = squareform(pz.load(data)).astype(np.int8)
        print "[*] Data matrix loaded."
        Y = np.array(pz.load(labels))
        print "[*] Data labels loaded."
        if m == "facts_matrix.pz":
            print "[*] Sampling matrix and labels..."
            fact_sample_size = 15000
            p = np.random.permutation(X.shape[0])[:fact_sample_size]
            X = X[p, :]
            X = X[:, p]
            Y = Y[p]
        print "Data: {}".format(data)
        print "Labels: {}".format(labels)
        map_range_evaluation(X, Y, fname)


def map_range_evaluation(X, Y, fname):
    """ Iterate over different values of distance b bits between
    hashes (0-65) and different values of k elements retrieved
    (5-50) and evaluate MAP for all combination of values.
    """
    results = []
    b_range = range(0, 65)
    k_range = range(5, 55, 5)

    widgets = ['Computing MAP: ',
               Percentage(), ' ',
               Bar(marker='#', left='[', right=']'),
               ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(b_range)*len(k_range))
    pbar.start()
    progress = 0
    for k in k_range:
        map_b_results = []
        for b in b_range:
            map_b_results.append(mapkb(X, Y, k, b))
            progress += 1
            pbar.update(progress)
        results.append(map_b_results)
    pbar.finish()

    pz.save(np.array(results), fname)
    print "Saved results: {}".format(fname)


def mapkb(X, Y, k=10, b=15):
    """ Compute MAP at k given a minimum distance in bits
    to consider to elements in similarity matrix X of
    the same class.
    """
    Y = np.array(Y)
    actual = []
    predicted = []
    for i, x in enumerate(X):
        idx_y_pred = np.where(x <= b)[0]
        y_pred = Y[idx_y_pred[np.argsort(x[idx_y_pred])]]
        actual.append([Y[i]])
        predicted.append(y_pred)
    return mapk(actual, predicted, k)


def run_family_experiment(families=[], simple=False):
    """ Compute MAP for each family given the overall optimal distance
    in bits between hashes. Save results in a different file for
    each type of data (facts, factlists, subgraphs)
    """
    print "Running MAP per family evaluation.."
    for m, l, fname in zip(MATRICES, LABELS, RESULTS):
        data = os.path.join(DATA_DIR, m)
        labels = os.path.join(DATA_DIR, l)
        map_results = pz.load(fname)
        X = squareform(pz.load(data)).astype(np.int8)
        Y = np.array(pz.load(labels))
        print "Data: {}".format(data)
        print "Labels: {}".format(labels)
        b_opt = np.argmax(map(np.max, map_results.T))
        print "b_opt: ", b_opt
        results = defaultdict(list)

        if families:
            for c in families:
                X_family = X[Y == c]
                for i, x in enumerate(X_family):
                    idx_y_pred = np.where(x <= b_opt)[0]
                    y_pred = Y[idx_y_pred[np.argsort(x[idx_y_pred])]]
                    results[c].append(y_pred)
            print "Computing MAP..."
            for actual, predicted in results.items():
                results[actual] = mapk([[actual]]*len(predicted),
                                       predicted, k=20)
            pz.save(results, fname.split('.')[0] + "_unique_families.pz")

        else:
            for i, x in enumerate(X):
                idx_y_pred = np.where(x <= b_opt)[0]
                y_pred = Y[idx_y_pred[np.argsort(x[idx_y_pred])]]
                results[Y[i]].append(y_pred)
            for actual, predicted in results.items():
                results[actual] = mapk([[actual]]*len(predicted),
                                       predicted, k=20)
            pz.save(results, fname.split('.')[0] + "_families.pz")


def run_family_simple_experiment(families=[]):

    print "Running simple MAP per family evaluation:"
    m, l, fname = MATRICES[2], LABELS[2], RESULTS[2]
    data = os.path.join(DATA_DIR, m)
    labels = os.path.join(DATA_DIR, l)
    print "Loading data..."
    X = squareform(pz.load(data)).astype(np.int8)
    Y = np.array(pz.load(labels))
    print "Data: {}".format(data)
    print "Labels: {}".format(labels)

    results = defaultdict(list)
    MAP = defaultdict(list)
    b_opt = 0
    for c in families:
        X_family = X[Y == c]
        for i, x in enumerate(X_family):
            idx_y_pred = np.where(x == b_opt)[0]
            y_pred = Y[idx_y_pred]
            np.random.shuffle(y_pred)
            results[c].append(y_pred)
    print "Computing MAP..."
    for actual, predicted in results.items():
        MAP[actual] = mapk([[actual]]*len(predicted), predicted, k=20)
    print MAP
    f_name = fname.split('.')[0] + "_unique_simple_families.pz"
    print "Saving file {}".format(f_name)
    pz.save(MAP, f_name)


def precision_recall_curves(families):
    """ Compute precision and recall curves for the families
    given as parameter. Each value is used over all and save
    """
    curves = {c: defaultdict(list) for c in families}
    for m, l, fname in zip(MATRICES, LABELS, RESULTS):
        data_type = m.split('_')[0]
        print "\nLoading {} data...".format(data_type)
        data = os.path.join(DATA_DIR, m)
        labels = os.path.join(DATA_DIR, l)
        X = squareform(pz.load(data)).astype(np.int8)
        Y = np.array(pz.load(labels))
        print "Data: {}".format(data)
        print "Labels: {}".format(labels)

        for c in families:
            print "Computing P-R curves for class \"{}\":".format(c)
            Y_family = label_binarize(Y, classes=[c]).flatten()
            X_family = X[Y_family == 1]
            # normalization
            X_family = 1 - X_family / float(X_family.max())
            widgets = ['',
                       Percentage(), ' ',
                       Bar(marker='#', left='[', right=']'),
                       ' ', ETA(), ' ']
            pbar = ProgressBar(widgets=widgets, maxval=X_family.shape[0])
            pbar.start()
            progress = 0
            for x in X_family:
                pr_curve = precision_recall_curve(Y_family, x)
                curves[c][data_type].append(pr_curve)
                progress += 1
                pbar.update(progress)
            pbar.finish()
    pz.save(curves, "results/precision_recall_curves.pz")


def precision_recall_simple_search(family, data_path):

    try:
        print "Loading X and Y..."
        X = pz.load("X.pz")
        Y = pz.load("Y.pz")
    except:
        families = pz.load("results/vt_families.pz")
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                 if f.endswith(".xml")]
        X = []
        Y = []
        widgets = ['Computing MAP: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(files))
        pbar.start()
        progress = 0
        for f in files:
            file_hash = os.path.basename(f).split('.')[0]
            try:
                file_family = families[file_hash]
                facts = get_facts_from_file(f)  # facts is an array
                X.append(facts)
                Y.append(file_family)
            except KeyError:
                continue
            except XMLSyntaxError:
                continue
            progress += 1
            pbar.update(progress)
        pbar.finish()
        Y = label_binarize(Y, [family]).flatten()
        pz.save(X, "X.pz")
        pz.save(Y, "Y.pz")

    X = np.array([np.array(x) for x in X])
    Y_idx = np.where(Y == 1)[0]
    true_facts = np.unique(np.hstack(X[Y_idx]))

    pr_values = []
    sample_len = 1000
    for iteration in range(10):
        true_sample_len = 5
        true_sample = random.sample(Y_idx, true_sample_len)
        sample = random.sample(range(X.shape[0]), sample_len - true_sample_len)
        sample_idx = np.unique(true_sample + sample)
        X_sample = X[sample_idx]
        facts_len = [x.shape[0] for x in X_sample]
        Y_sample = np.zeros(sum(facts_len), dtype=bool)
        i = 0
        for j, idx in enumerate(sample_idx):
            l = facts_len[j]
            if Y[idx]:
                Y_sample[i:i+l] = True
            i += l

        # X_sample is an array of 10000 samples x
        # Y_sample is a bool array of 10000 * len(x)

        s = 0
        index_of_x_in_y = []
        for i, fl in enumerate(facts_len):
            index_of_x_in_y.append(s)
            s += facts_len[i]

        predicted = np.zeros(Y_sample.shape[0], dtype=bool)
        widgets = ['Matching facts: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=true_facts.shape[0])
        pbar.start()
        progress = 0
        for k, fact in enumerate(true_facts):
            for l, x in enumerate(X_sample):
                x_true_idx = np.where(x == fact)[0] + index_of_x_in_y[l]
                predicted[x_true_idx] = True

            precision = precision_score(Y_sample, predicted)
            recall = recall_score(Y_sample, predicted)
            pr_values.append((precision, recall))
            predicted[:] = False
            progress += 1
            pbar.update(progress)
        pbar.finish()
        pz.save(pr_values, "results/pr_values_simple_search_{}.pz".format(family))


def get_facts_from_file(f):
    return [i for i in etree.fromstring(open(f).read()).itertext()
            if '\n' not in i]


def print_precision_recall_simple_search(family=[]):
    print "\nAverage Precision & Recall for Exact Fact Matching Search"
    for f in family:
        pr, re = zip(*set(pz.load('results/pr_values_simple_search_{}.pz'.format(f))))
        print ""
        print "Family: {}".format(f.upper())
        print "Avg. Precision: {}".format(np.average(pr))
        print "Avg. Recall: {}".format(np.average(re))


if __name__ == "__main__":

    run_experiment()
    run_family_experiment()
    run_family_experiment(['stuxnet', 'regin'])
    run_family_simple_experiment(['stuxnet', 'regin'])
    precision_recall_curves(['stuxnet', 'regin'])
    precision_recall_simple_search('stuxnet')
    precision_recall_simple_search('regin')
    print_precision_recall_simple_search(['stuxnet', 'regin'])
    pass
