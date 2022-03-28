import yaml
import csv
import os
import codecs
from collections import Counter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, "r")))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, "a") as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def eval_bleu(reference, candidate, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip("/")
    ref_file = log_path + "/reference.txt"
    cand_file = log_path + "/candidate.txt"
    with codecs.open(ref_file, "w", "utf-8") as f:
        for s in reference:
            f.write(" ".join(s) + "\n")
    with codecs.open(cand_file, "w", "utf-8") as f:
        for s in candidate:
            f.write(" ".join(s).strip() + "\n")

    temp = log_path + "/result.txt"
    command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)

    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)

    try:
        bleu = float(result.split(",")[0][7:])
    except ValueError:
        bleu = 0

    return result, bleu


def eval_multi_bleu(references, candidate, log_path):
    ref_1, ref_2, ref_3, ref_4 = [], [], [], []
    for refs, cand in zip(references, candidate):
        ref_1.append(refs[0])
        if len(refs) > 1:
            ref_2.append(refs[1])
        else:
            ref_2.append([])
        if len(refs) > 2:
            ref_3.append(refs[2])
        else:
            ref_3.append([])
        if len(refs) > 3:
            ref_4.append(refs[3])
        else:
            ref_4.append([])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip("/")
    ref_file_1 = log_path + "/reference_1.txt"
    ref_file_2 = log_path + "/reference_2.txt"
    ref_file_3 = log_path + "/reference_3.txt"
    ref_file_4 = log_path + "/reference_4.txt"
    cand_file = log_path + "/candidate.txt"
    with codecs.open(ref_file_1, "w", "utf-8") as f:
        for s in ref_1:
            f.write(" ".join(s) + "\n")
    with codecs.open(ref_file_2, "w", "utf-8") as f:
        for s in ref_2:
            f.write(" ".join(s) + "\n")
    with codecs.open(ref_file_3, "w", "utf-8") as f:
        for s in ref_3:
            f.write(" ".join(s) + "\n")
    with codecs.open(ref_file_4, "w", "utf-8") as f:
        for s in ref_4:
            f.write(" ".join(s) + "\n")
    with codecs.open(cand_file, "w", "utf-8") as f:
        for s in candidate:
            f.write(" ".join(s).strip() + "\n")

    temp = log_path + "/result.txt"
    command = (
        "perl multi-bleu.perl "
        + ref_file_1
        + " "
        + ref_file_2
        + " "
        + ref_file_3
        + " "
        + ref_file_4
        + "<"
        + cand_file
        + "> "
        + temp
    )
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        bleu = float(result.split(",")[0][7:])
    except ValueError:
        bleu = 0
    return result, bleu


def bleu_compute(reference, candidate, n):
    return modified_precision(reference, candidate, n)


def modified_precision(reference, candidate, n):
    counts = Counter(ngrams(candidate, n)) if len(candidate) >= n else Counter()
    max_counts = {}
    for ref in reference:
        ref_counts = (
            Counter(ngrams(ref, n)) if len(ref) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts[ngram])

    clipped_counts = {
        ngram:min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))
    return numerator / denominator