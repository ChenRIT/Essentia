#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
import spacy
from monolingualWordAligner.wordAligner import Aligner

from generate_word_alignment import make_alignment_matrix, get_align_indices_by_rule, get_align_indices_sultan

nlp = spacy.load("en_core_web_lg")
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def split_crisscross(alignments, sents_dic={}, to_merge=None):
    #print("length of alignments: {}".format(len(alignments)))
    groups = []
    if to_merge is None:
        to_merge = range(len(alignments))
    good_set = set()
    bad_set = set()
    for i in to_merge:
        consistent = True
        for j in good_set:
            # this sorts alignments based on first entry
            idx1 = i
            idx2 = j
            if i < j:
                idx1 = j
                idx2 = i
            #sorted_alignment = sorted(alignments[idx1][idx2])
            sorted_alignment = sorted(get_align_indices_sultan(alignments, idx1, idx2))
            second_entries = list(x[1] for x in sorted_alignment)
            # the second entry should be ascending if there is no crisscross
            if sorted(second_entries) != second_entries:
                consistent = False
                # print("Crisscross between sent {} and sent {}:".format(i, j))                
                # print("Alignment causing crisscross: {}".format(alignments[idx1][idx2]))                
                break
        if consistent:
            good_set.add(i)
        else:
            logging.debug("crisscross found between sent {} and sent {}.".format(i, j))
            logging.debug("Sent {}: {}".format(i, sents_dic.get(i)))
            logging.debug("Sent {}: {}".format(j, sents_dic.get(j)))            
            bad_set.add(i)
    if len(bad_set) > 0:
        groups = split_crisscross(alignments, sents_dic, list(sorted(bad_set)))
        return [list(good_set)] + groups
    else:
        return [list(good_set)]


def split_none_injective(alignments, to_merge=None):
    #print("Alignments: {}".format(alignments))
    groups = []
    if to_merge is None:
        to_merge = range(len(alignments))
    good_set = set()
    bad_set = set()
    for i in to_merge:
        consistent = True
        for j in good_set:
            # this sorts alignments based on first entry
            idx1 = i
            idx2 = j
            if i < j:
                idx1 = j
                idx2 = i
            #sorted_alignment = sorted(alignments[idx1][idx2])
            sorted_alignment = None
            sorted_alignment = sorted(get_align_indices_sultan(alignments, idx1, idx2))
            first_entries = list(x[0] for x in sorted_alignment)
            second_entries = list(x[1] for x in sorted_alignment)
            # is it one to many?
            if len(first_entries) > len(set(first_entries)):
                consistent = False
            # is it many to one?
            if len(second_entries) > len(set(second_entries)):
                consistent = False
        if consistent:
            good_set.add(i)
        else:
            logging.debug("Injectiveness found between sent {} and sent {}.".format(i, j))            
            #print("Injectiveness between sent {} and sent {}:".format(i, j))
            #print("Alignment causing injectiveness: {}".format(alignments[idx1][idx2]))
            bad_set.add(i)
    if len(bad_set) > 0:
        groups = split_none_injective(alignments, list(sorted(bad_set)))
        return [list(good_set)] + groups
    else:
        return [list(good_set)]


def transitive(alignments, i, j, k, sents_dic):
    #print("sents_dic: {}".format(sents_dic))
    perms = itertools.permutations([i, j, k])
    for (n1, n2, n3) in perms:
        if n1 > n2:
            #dict_12 = dict(alignments[n1][n2])
            #dict_12 = dict(get_align_indices(alignments, n1, n2))
            dict_12 = None
            dict_12 = dict(get_align_indices_sultan(alignments, n1, n2))
        else:
            #dict_12 = dict((y, x) for (x, y) in alignments[n2][n1])
            #dict_12 = dict((y, x) for (x, y) in get_align_indices(alignments, n2, n1))
            dict_12 = None
            dict_12 = dict((y, x) for (x, y) in get_align_indices_sultan(alignments, n2, n1))
        if n2 > n3:
            #dict_23 = dict(alignments[n2][n3])
            #dict_23 = dict(get_align_indices(alignments, n2, n3))
            dict_23 = None
            dict_23 = dict(get_align_indices_sultan(alignments, n2, n3))
        else:
            #dict_23 = dict((y, x) for (x, y) in alignments[n3][n2])
            #dict_23 = dict((y, x) for (x, y) in get_align_indices(alignments, n3, n2))
            dict_23 = None
            dict_23 = dict((y, x) for (x, y) in get_align_indices_sultan(alignments, n3, n2))
        if n1 > n3:
            #dict_13 = dict(alignments[n1][n3])
            #dict_13 = dict(get_align_indices(alignments, n1, n3))
            dict_13 = None
            dict_13 = dict(get_align_indices_sultan(alignments, n1, n3))
        else:
            #dict_13 = dict((y, x) for (x, y) in alignments[n3][n1])
            #dict_13 = dict((y, x) for (x, y) in get_align_indices(alignments, n3, n1))
            dict_13 = None
            dict_13 = dict((y, x) for (x, y) in get_align_indices_sultan(alignments, n3, n1))
    for s, t in dict_12.items():
        t2 = dict_23.get(t, None)
        if t2 is not None and t2 != dict_13.get(s, None):
            logging.debug("Transitivity is violated for the following sentences:")
            # logging.debug("Sent {}: {}".format(i, sents_dic[i]))
            # logging.debug("Sent {}: {}".format(j, sents_dic[j]))
            # logging.debug("Sent {}: {}".format(k, sents_dic[k]))
            if i > j:
                logging.debug("Alignment between sent {} and sent {}: {}".format(i, j, alignments[i][j]))
            else:
                logging.debug("Alignment between sent {} and sent {}: {}".format(j, i, alignments[j][i]))
            if j > k:
                logging.debug("Alignment between sent {} and sent {}: {}".format(j, k, alignments[j][k]))
            else:
                logging.debug("Alignment between sent {} and sent {}: {}".format(k, j, alignments[k][j]))
            if i > k:
                logging.debug("Alignment between sent {} and sent {}: {}".format(i, k, alignments[i][k]))
            else:
                logging.debug("Alignment between sent {} and sent {}: {}".format(k, i, alignments[k][i]))
            return False
    return True


def split_none_transitive(alignments, sents_dic, to_merge=None):
    groups = []
    if to_merge is None:
        to_merge = range(len(alignments))
    good_set = set()
    bad_set = set()
    for i in to_merge:
        consistent = True
        for j in good_set:
            for k in good_set:
                if k >= j: continue
                # checking if it is transitive
                if not transitive(alignments, i, j, k, sents_dic):
                    consistent = False
        if consistent:
            good_set.add(i)
        else:
            bad_set.add(i)
    if len(bad_set) > 0:
        groups = split_none_transitive(alignments, sents_dic, list(sorted(bad_set)))
        return [list(good_set)] + groups
    else:
        return [list(good_set)]


def create_valid_groups(alignments, sents_dic):
    #print("Alignment during validity checking: {}".format(alignments))
    all_groups = []
    groups1 = split_none_injective(alignments)
    #print("groups1:{}".format(groups1))
    for g in groups1:
        groups2 = None
        groups2 = split_crisscross(alignments, sents_dic, g)
        #print("\t groups2: {}".format(groups2))
        for g2 in groups2:
            #print("\t\t groups3: {}".format(split_none_transitive(alignments, g2)))
            all_groups += split_none_transitive(alignments, sents_dic, g2)
                
    return all_groups


def normalize_sent(sent):
    """
    Normalize numbers and named entities in a sentence
    """
    logging.debug("Normalizing the sentence: {}".format(sent))
    doc = nlp(sent)

    # Normalize numbers
    norm_num_tks = ["NUM" if tk.like_num else tk.text for tk in doc]
    #print("norm_num_tks: {}".format(norm_num_tks))
    new_sent = " ".join(norm_num_tks)

    # Normalize entities
    new_doc = nlp(new_sent)
    norm_ent_sent = ""
    last_index = 0
    for ent in new_doc.ents:
        if ent.lower_ == "wifi":
            continue # A hack to ty datasets
        
        #print("Entity: {}".format(ent.text))
        if ent.text == "NUM":
            continue
        
        start_idx = ent.start_char
        end_idx = ent.end_char
        label = ent.label_
        #print("start_char: {}".format(ent.start_char))
        #print("end_char: {}".format(ent.end_char))
        
        norm_ent_sent += new_sent[last_index:start_idx]
        #print("non-ent span: {}".format(new_sent[last_index:start_idx]))

        norm_ent_sent += label
        last_index = end_idx

    if last_index != len(new_sent):
        norm_ent_sent += new_sent[last_index:]

    logging.debug("Normalized sent: {}".format(norm_ent_sent))
    return norm_ent_sent


def divide_sent_by_prep(sent):
    """
    Divide a sentence into several chunks by the signal of preposition
    """
    logging.info("Sent to chunk: {}".format(sent))
    doc = nlp(sent)
    chunks = []
    last_idx = 0
    for tk in doc:
        if tk.pos_ == "ADP":
            cur_idx = tk.i
            past_chunk = doc[last_idx:cur_idx].text
            chunks.append(past_chunk)
            last_idx = cur_idx

    last_chunk = doc[last_idx:].text
    chunks.append(last_chunk)
    logging.debug("Chunking results: {}".format(chunks))

    return chunks

def main():
    # small test for the preprocess functions
    # I assume the input contains: list of lists storing alignments
    sents = ['This is simply a test.',
             'is this simply a test?',
             'simply a test this is.']
    alignments = make_alignment_matrix(sents)
    # checking if we detect criss-cross
    print(split_crisscross(alignments))
    # checking if we detect many-to-1s
    # creating a none injective mapping
    #alignments[2][0][2][1] = 3
    print(split_none_injective(alignments))
    # checking if we detect non-transitives
    # creating some non-transitive issue
    #alignments[2][0][1][1] = 4
    print(split_none_transitive(alignments))
    # checking all conditions together
    print(create_valid_groups(alignments))

    
def test_norm():
    test_sent1 = u"Directions to Tulsa with road 66"
    test_sent2 = u"Show me the way to go to 33 Greene Street"

    normalize_sent(test_sent1)
    normalize_sent(test_sent2)

def test_sent_chunk():
    test_sent1 = u"Directions to Tulsa with road 66"
    test_sent2 = u"Show me the way to go to 33 Greene Street"

    divide_sent_by_prep(test_sent1)
    divide_sent_by_prep(test_sent2)
    
    
if __name__ == '__main__':
    # main()
    # test_norm()
    test_sent_chunk()
