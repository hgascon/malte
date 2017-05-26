# MALTE
# Evaluation functions for experiment results
# (c) 2017 Hugo Gascon

import operator
import json
import pz
import re
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import sys

VT_FAMILIES = "vt_families.pz"
VT_CLUSTERS = "vt_clusters.pz"
DELIMITERS = [".", "/", " ", "-", "[", "]", ";", ":", "!", "(", ")", "_",
              "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
INVALID_LABELS = ["trojan", "adware", "android", "virus",
                  "worm", "variant", "generic", "agent",
                  "html", "application", "unwanted", "troj",
                  "malware", "behaveslike", "exploit", "small",
                  "program", "suspicious", "inject", "heuristic",
                  "potentially", "reputation", "suspected",
                  "riskware", "spyware", "multi", "generikcd"]


def tokenize(s):
    return s.split(',')


def find_clusters(results, clusters=50):
    """
    Build a bag of tokens model from the labels assigned
    to each sample by different AVs and generate
    clusters based on the vector representation.

    :param results: dict (hash: AV labels dict)
    :return:
    """
    hashes = []
    x = []
    regex = '|'.join(map(re.escape, DELIMITERS))
    for h, report in results.items():
        hashes.append(h)
        labels = []
        for av_name, label in report.items():
            label = label.lower()
            labels += re.split(regex, label)
        labels = ','.join([l for l in labels if len(l) > 2])
        x.append(labels)

    # 2. vectorize each report with the set of strings
    vect = CountVectorizer(tokenizer=tokenize)
    X = vect.fit_transform(x)
    X = X/X.max()

    # 3. find clausters for the reports according to the av labels
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    y = kmeans.predict(X)

    return dict(zip(hashes, y))


def get_family_majority_voting(report):
    """
    Returns the probable name of a malware sample after
    removing generic words and doing majority voting
    of the AV labels.

    :param report: dict (AV name: detection/label)
    :return: str (family name)
    """
    regex = '|'.join(map(re.escape, DELIMITERS))
    labels = []
    for av_name, label in report.items():
        label = label.lower()
        labels += re.split(regex, label)
    c = Counter(labels)
    sorted_c = sorted(c.items(), key=operator.itemgetter(1))
    sorted_c.reverse()
    families = []
    for i, label_freq in enumerate(sorted_c):
        if len(label_freq[0]) > 4:
            families.append(label_freq)
    while families[0][0] in INVALID_LABELS:
        families = families[1:]
    return families[0][0]


def normalize_name(name):
    if name == "bdr":
        return "backdoor"
    return name


def find_family_names(results):
    """

    :param results: dict (hash: AV labels dict)
    :return: dict (hash: family name)
    """
    families = {}
    for h, report in results.items():
        try:
            family = get_family_majority_voting(report)
            families[h] = normalize_name(family)
        except:
            pass

    c = Counter(families.values())
    for k, v in c.items():
        if v < 40:
            c.pop(k)
    valid_families = c.keys()

    for h, family in families.items():
        if family not in valid_families:
            families.pop(h)

    return families


def load_json_results_vt(filename):
    """
    Load a virus total detections file and
    generate a dictionary of hashes

    :return: dict (hash: AV labels dict)
    """
    results = {}
    file = open(filename).read()
    for i, line in enumerate(file.split('\n')):
        try:
            h, av_labels = line.split('\t')
        except ValueError as e:
            print "{} in line {}".format(e, i)
            pass
        results[h] = json.loads(av_labels)
    return results


def manual_labeling():
    family = []
    VT_results = pz.load("vt_results.pz")
    for analysis in VT_results:
        md5 = analysis['md5']
        names = []
        for av, result in analysis['scans'].items():
            if result['detected']:
                names.append(result['result'])
        for n in names:
            print n
        print
        families = list(set([i[1] for i in family]))
        for i, f in enumerate(families):
            print i, f
        print
        name = raw_input("Select a name for the sample with md5 {}: ".format(md5))
        print
        try:
            family.append([md5, families[int(name)]])
        except:
            family.append([md5, name])

    pz.save(family, "family_manual.pz")


if __name__ == "__main__":

    print "[*] Loading VT detections..."
    results = load_json_results_vt(sys.argv[1])
    print "[*] Finding family names..."
    families = find_family_names(results)
    pz.save(families, VT_FAMILIES)
    print "File saved {}.".format(VT_FAMILIES)
    print
    print "[*] Finding cluster names..."
    clusters = find_clusters(results)
    pz.save(clusters, VT_CLUSTERS)
    print "File saved {}.".format(VT_CLUSTERS)
    print "Done."
