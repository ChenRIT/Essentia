# Mine domain-specific paraphrases from SNIPs datasets
import os
import networkx as nx
import matplotlib.pyplot as plt
import logging
import csv
import spacy
import codecs
from generate_word_alignment import make_alignment_matrix
from preprocessing import create_valid_groups, normalize_sent, divide_sent_by_prep
from fsa import create_fsa, process_sents, find_verb_paraphrases, find_phrase_paraphrases, idx_to_node, get_node_text, display_graph, get_repr_nd_text

nlp = spacy.load('en_core_web_lg')

def create_fsa_from_file(fpath):
    """
    Generate all sentence-level paraphrases for a given data file
    """
    # Load sentences
    sents = []
    tk_sents = {}
    with codecs.open(fpath, encoding='utf-8') as ifile:
        idx = 0
        for line in ifile:
            line = line.strip()
            norm_line = normalize_sent(line)
            if norm_line in sents:
                print("Found a duplicated sentence.")
                continue
            
            doc = nlp(norm_line)
            sent_tks = [tk.text for tk in doc]
            
            tk_sents[idx] = sent_tks
            #print("Sent {}: {}".format(idx, line))
            #print("Normalized sent: {}".format(idx, norm_line))            
            sents.append(norm_line)
            idx += 1
    
    # Create word alignment
    align_matrix = make_alignment_matrix(sents)

    # Validity checking
    sents_cluster = create_valid_groups(align_matrix, tk_sents)
    #print("Sent clusters: {}".format(sents_cluster))
    
    # Create the word lattice
    fsa = create_fsa(tk_sents)
    for i, cluster in enumerate(sents_cluster):
        fsa = process_sents(fsa, tk_sents, align_matrix, cluster)

    sent_num = len(sents)
    tk_num = 0
    for _, tk_list in tk_sents.items():
        tk_num += len(tk_list)

    # Display the word lattice
    # print("idx_to_node:{}".format(idx_to_node))
    # nx.draw_circular(fsa, with_labels=True)
    # plt.show()
    
    return fsa, sent_num, sents, tk_num


def store_paraphrase_plain_text(word_to_para, output_dir):
    logging.info("Store paraphrases in plain text.")
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0]+"_essentia.txt")
    
    with open(output_file, 'w') as ofile:
        for ndpair, para_set in word_to_para.items():
            nd1 = ndpair[0]
            nd2 = ndpair[1]
            nd1_text = get_node_text(fsa, nd1)
            nd2_text = get_node_text(fsa, nd2)
            ofile.write("\n\n")
            ofile.write("node 1: " + str(nd1_text) + "\n")
            ofile.write("node 2: " + str(nd2_text) + "\n")            
            for phrase in para_set:
                ofile.write(phrase)
                ofile.write("\n")
    # print("All sentences: {}".format(all_sents))

    
def store_paraphrase_csv(input_path, fsa, word_to_para, output_dir):
    logging.info("Store paraphrases in csv files.")
    dataset_name = os.path.basename(input_path)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0]+"_essentia.csv")
    
    with open(output_file, "w") as ofile:
        csv_writer = csv.writer(ofile)
        for ndpair, para_set in word_to_para.items():
            nd1 = ndpair[0]
            nd2 = ndpair[1]
            nd1_text = get_repr_nd_text(fsa, nd1)
            nd2_text = get_repr_nd_text(fsa, nd2)

            csv_writer.writerow([dataset_name, nd1_text, para_set, nd2_text])
            
            # End loop

def store_vp_paraphrase_csv(input_path, fsa, word_to_para, output_dir):
    logging.info("Store vp paraphrases in csv files.")
    dataset_name = os.path.basename(input_path)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0]+"_essentia.csv")

    # Merge paraphrases of different clusters together in word_to_para
    para_sets = [paras for _, paras in word_to_para.items() if len(paras) > 0]
    #print("para_sets: {}".format(para_sets))
    set_to_root = {} # A union find data structure
    for i in range(len(para_sets)):
        set_to_root[i] = i
        
    for i, set1 in enumerate(para_sets):
        if i == len(para_sets) - 1:
            break
        
        for j in range(i+1, len(para_sets)):
            #print("i set: {}".format(set1))
            #print("j set: {}".format(para_sets[j]))
            has_dup = False
            for ele in set1:
                if ele in para_sets[j]:
                    has_dup = True
                    break

            if has_dup:
                #print("set {} and set {} share elements.".format(i, j))
                set_to_root[j] = set_to_root[i]
    #print("set_to_root: {}".format(set_to_root))

    root_to_mems = {} # all paraphrase sets of the same member
    for key, value in set_to_root.items():
        equi_sets = root_to_mems.get(value)
        #print("equi_sets: {}".format(equi_sets))
        if equi_sets is None:
            root_to_mems[value] = [key]
            continue

        new_sets = list(equi_sets)
        new_sets.append(key)
        #print("new_sets: {}".format(new_sets))
        root_to_mems[value] = new_sets
    #print("root_to_mems: {}".format(root_to_mems))

    new_para_sets = []
    for _, equi_sets_ids in root_to_mems.items():
        equi_set = []
        for id in equi_sets_ids:
            equi_set += para_sets[id]
        new_para_sets.append(list(set(equi_set)))
    
    with open(output_file, "w") as ofile:
        csv_writer = csv.writer(ofile)
        for para_set in new_para_sets:
            if len(para_set) > 0:
                csv_writer.writerow([dataset_name, para_set])
            
            # End loop
        
    # with open(output_file, "w") as ofile:
    #     csv_writer = csv.writer(ofile)
    #     for ndpair, para_set in word_to_para.items():
    #         if len(para_set) > 0:
    #             csv_writer.writerow([dataset_name, para_set])
            
    #         if one_example_mode:
    #             break
    #         # End loop
            
def gene_phrase_paras_single_file(input_file, output_dir):
    logging.info("Generate phrase paraphrases for: {}".format(input_file))

    fsa, sent_num = create_fsa_from_file(input_file)
    word_to_para = find_phrase_paraphrases(fsa)

    # Store paraphrases
    #store_paraphrase_plain_text()
    store_paraphrase_csv(input_file, fsa, word_to_para, output_dir)

    
def gene_phrase_paras_first_chunk_single_file(input_file, output_dir, origin_sents):
    logging.info("Select the first chunk of each sentence as paraphrases: {}".format(input_file))

    paras = []
    with open(input_file) as ifile:
        for line in ifile:
            chunks = divide_sent_by_prep(unicode(line, "utf-8"))
            paras.append(chunks[0])
    paras = list(set(paras))

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0]+"_essentia.txt")
    with open(output_file, "w") as ofile:
        for p in paras:
            ofile.write(p)
            ofile.write("\n")
            
    return paras


def gene_phrase_paras_vps_single_file(input_file, output_dir):
    logging.info("Generate verb phrase paraphrases for: {}".format(input_file))

    fsa, sent_num, origin_sents, _ = create_fsa_from_file(input_file)
    # Find verb phrases out of the paraphrases    
    word_to_para = find_verb_paraphrases(fsa, origin_sents)

    # Store paraphrases
    #store_paraphrase_plain_text()
    store_vp_paraphrase_csv(input_file, fsa, word_to_para, output_dir)
    print("\n\n Candidate paraphrases can be found at: {} \n\n".format(output_dir))

def gene_para_dir_first_chunk(output_dir):
    snips_dir = "datasets/processed/snips"
    for input_file in os.listdir(snips_dir):
        fpath = os.path.join(snips_dir, input_file)
        print(fpath)
        gene_phrase_paras_first_chunk_single_file(fpath, output_dir)

        
def gene_para_dir(output_dir):
    snips_dir = "datasets/processed/snips"
    for input_file in os.listdir(snips_dir):
        fpath = os.path.join(snips_dir, input_file)
        print(fpath)
        #gene_phrase_paraphrases(fpath)
        gene_phrase_paras_single_file(fpath, output_dir)

def gene_vp_para_dir(output_dir, input_dir):
    for input_file in os.listdir(input_dir):
        fpath = os.path.join(input_dir, input_file)
        print("Processing {} ...".format(fpath))
        #gene_phrase_paraphrases(fpath)
        gene_phrase_paras_vps_single_file(fpath, output_dir)

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_dir = "./datasets/"
    output_dir = "./snips_paras/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    gene_vp_para_dir(output_dir, input_dir)
