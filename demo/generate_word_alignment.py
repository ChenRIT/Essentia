#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import spacy
import itertools
from monolingualWordAligner.wordAligner import Aligner

nlp = spacy.load('en_core_web_lg')

def load_file(filepath):
    with open(filepath) as ofile:
        return [unicode(x.rstrip()) for x in ofile.readlines()]

def get_align_indices_by_rule(align_matrix, sent_idx1, sent_idx2):
    if sent_idx1 == sent_idx2:
        logging.warning("No self alignment available!")
        return []
    
    large_idx = sent_idx1
    small_idx = sent_idx2
    if sent_idx1 < sent_idx2:
        large_idx = sent_idx2
        small_idx = sent_idx1

    alignment_list = align_matrix[large_idx][small_idx]
    print("Alignment list: {}".format(alignment_list))
    align_indices = [ele[0] for ele in alignment_list]
    print("Alignment indices: {}".format(align_indices))

    #sys.exit()
    return align_indices

def get_align_indices_sultan(align_matrix, sent_idx1, sent_idx2):
    if sent_idx1 == sent_idx2:
        logging.warning("No self alignment available!")
        return []
    
    large_idx = sent_idx1
    small_idx = sent_idx2
    if sent_idx1 < sent_idx2:
        large_idx = sent_idx2
        small_idx = sent_idx1

    alignment_list = align_matrix[large_idx][small_idx]

    return alignment_list


def make_alignment_matrix(sents):
    aligner = Aligner('spacy')
    num_sents = len(sents)
    alignments = [[] for x in range(num_sents)]
    for i, sent1 in enumerate(sents):
        for j, sent2 in enumerate(sents):
            if j >= i:
                continue

            align_res = aligner.align_sentences(sent1, sent2)
            alignments[i].append(align_res)
            # print("Sent1: {}".format(sent1))
            # print("Sent2: {}".format(sent2))            
            # print("Alignment: {}".format(align_res))
            #sys.exit()
    return alignments

def make_alignment_matrix_with_rules(sents):
    num_sents = len(sents)
    alignments = [[] for x in range(num_sents)]
    for i, sent1 in enumerate(sents):
        for j, sent2 in enumerate(sents):
            if j >= i:
                continue

            align_res = align_words(sent1, sent2)
            adjust_alignment = []
            for ele in align_res:
                idx_pair = ele[0]
                tk_pair = ele[1]
                new_idx_pair = [idx_pair[0]+1, idx_pair[1]+1]
                adjust_alignment.append((new_idx_pair, tk_pair))
            # print("Sent {}: {}".format(i, sent1))
            # print("Sent {}: {}".format(j, sent2))            
            # print("Adjusted alignment: {}".format(adjust_alignment))
            alignments[i].append(adjust_alignment)
    #print("Alignment during creation: {}".format(alignments))
    return alignments


def create_align_matrix(sents):
    # Align words in sentences
    align = {}
    for i in range(len(sents)):
        for j in range(0, i):
            sent1 = sents[i]
            sent2 = sents[j]
            print("Aligning sents:\n{}\n{}".format(sent1, sent2))
            align_indices = processing.align_sentences(sent1,sent2)
            align[(i, j)] = align_indices

    return align

def make_spacy_docs(sents):
    nlp = spacy.load('en')
    return [nlp(x) for x in sents]


def find_chunks(doc):
    chunks = []
    for np in doc.noun_chunks:
        chunks.append(np)

def merge_vp(align_matrix, sentence):
    """
    Merge verb chunks for each sentence and update the alignment within a cluster
    """
    pattern = r'<VERB>?<ADV>*<VERB>+'
    doc = textacy.Doc(sentence, lang='en_core_web_lg')
    cand_lists = textacy.extract.pos_regex_matches(doc, pattern)
    print("Verb phrases list: {}".format(cand_lists))    
    # vp = []
    # for list in cand_lists:
    #     vp.append(list.text)
    # return vp
        
def merge_np(align_matrix, sents_dic, orig_sents):
    """
    Merge noun chunks for each sentence and update the alignment within a cluster
    """
    for id in sents_dic:
        sent = orig_sents[id]
        # doc = nlp(unicode(sent, 'utf-8'))
        doc = nlp(sent)

        # Compute the indices mapping between tokens, and phrases after merging noun phrases.
        idx_map = {}
        cur_idx_tk = 0 # The index for the tokenized sentence
        cur_idx_chunk = 0 # The index for the sentence with noun chunks merged
        print("Noun chunks for sent {}: {}".format(sent, [ck.text for ck in doc.noun_chunks]))
        for chunk in doc.noun_chunks:
            chunk_start = chunk.start
            chunk_end = chunk.end
            # print("Chunk starts at {} and ends at {}".format(chunk_start, chunk_end))
            for i in range(cur_idx_tk, chunk_start):
                # print("Map non-chunk {} to {}".format(i, cur_idx_chunk))
                idx_map[i] = cur_idx_chunk
                cur_idx_chunk += 1
            for i in range(chunk_start, chunk_end):
                cur_idx_tk = i
                # print("Map within-chunk {} to {}".format(i, cur_idx_chunk))                
                idx_map[i] = cur_idx_chunk
            cur_idx_tk += 1
            cur_idx_chunk += 1
        for i in range(cur_idx_tk, len(doc)):
            # print("Map remaining {} to {}".format(i, cur_idx_chunk))            
            idx_map[i] = cur_idx_chunk
            cur_idx_chunk += 1

        #print("idx_map:{}".format(idx_map))
            
        # Use idx_map to revise the align_matrix
        for id2 in sents_dic:
            if id2 == id:
                continue
            if id2 < id:
                align_list = find_align_element(align_matrix, id, id2)
                new_align_list = []
                for ele in align_list:
                    new_ele = list(ele)
                    new_ele[0] = idx_map[new_ele[0]-1]+1
                    new_align_list.append(new_ele)
                unique_align_list = [k for k,_ in itertools.groupby(sorted(new_align_list))]
                update_align_element(align_matrix, id, id2, unique_align_list)
            else:
                align_list = find_align_element(align_matrix, id2, id)
                new_align_list = []
                for ele in align_list:
                    new_ele = list(ele)
                    new_ele[1] = idx_map[new_ele[1]-1]+1
                    new_align_list.append(new_ele)
                unique_align_list = [k for k,_ in itertools.groupby(sorted(new_align_list))]
                update_align_element(align_matrix, id2, id, unique_align_list)

        # Update the sentence in sents_dic
        new_sent = []
        phrase = ''
        last_chunk_id = 0
        #print("sents update: {}".format(sents_dic[id]))
        #print("idx_map: {}".format(idx_map))
        for i, word in enumerate(sents_dic[id]):
            chunk_id = idx_map[i]
            if chunk_id == last_chunk_id:
                if phrase == '':
                    phrase = word
                else:
                    phrase = ' '.join([phrase, word])
            else:
                new_sent.append(phrase)
                phrase = word
            last_chunk_id = chunk_id
        new_sent.append(phrase)
        sents_dic[id] = new_sent
        #print("Old sent: {}".format(sents_dic[id]))
        #print("New sent: {}".format(new_sent))
        
def merge_chunks(align_matrix, sents_dic, orig_sents):
    merge_np(align_matrix, sents_dic, orig_sents)
    #merge_vp(align_matrix, sents_dic, orig_sents, sent_ids)


def find_align_element(align_matrix, i, j):
    return get_align_indices_sultan(align_matrix, i, j)


def main():
    sents = load_file(sys.argv[1])
    # alignments = make_alignment_matrix(sents)
    docs = make_spacy_docs(sents)
    print(find_chunks(docs[0]))


if __name__ == '__main__':
    main()
