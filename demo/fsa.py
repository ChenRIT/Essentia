import networkx as nx
from networkx.algorithms.dag import descendants
import ast
from generate_word_alignment import *
from utility import merge_dics, find_dobj_phrase, find_prep_pobj_phrase

idx_to_node = {} # {nd_id: {sent_id: [tk_id, tk]}}
node_to_idx = {} # Node contents to node indices mapping
# The content of a node is a dictionary: {sent_id: [token_id, token_text]}
start_state = -1
end_state = -2
idx_to_node[start_state] = "-1" # Start node
idx_to_node[end_state] = "-2" # End node
globe_node_idx = 2
align_threshold = float(1) / 3


def init_fsa():
    fsa = nx.DiGraph()
    fsa.add_node(start_state)
    fsa.add_node(end_state)
    return fsa

def create_fsa(all_sents_dic):
    fsa = init_fsa()
    for id in all_sents_dic:
        fsa = add_sent_fsa(fsa, all_sents_dic[id], id)
    return fsa


def find_node(fsa, sent_id, wd_id):
    """
    Find a node representing the wd_id-th word of a sentence of sent_id
    """
    node = None
    for nd in fsa:
        if nd == start_state or nd == end_state:
            continue

        node_content = idx_to_node.get(int(nd))
        if node_content is None:
            print("Node content of node {} is None".format(nd))
            break

        #print("Node content of node {}: {}".format(nd, node_content))
        nd_dic = ast.literal_eval(str(node_content))
        label_sent = nd_dic.get(sent_id)
        if label_sent is None:
            continue
        else:
            if label_sent[0] == wd_id:
                node = nd
                break
            else:
                continue

    if node is None:
        print("Node of {} not found!\n".format((sent_id, wd_id)))
        find_nodes_of_sent(fsa, sent_id)

    #print("The node found: {}".format(node))
    return node

def find_nodes_of_sent(fsa, sent_id):
    """
    Find all the nodes of a given sentence
    """
    labels = []
    for nd in fsa:
        if nd == start_state or nd == end_state:
            continue

        node_content = idx_to_node.get(int(nd))
        if node_content is None:
            print("Node content of node {} is None".format(nd))
            break
        
        nd_dic = ast.literal_eval(str(node_content))
        label_sent = nd_dic.get(sent_id)
        if label_sent is None:
            continue
        else:
            labels.append(label_sent)
    print("All labels of sent {}:\n {}".format(sent_id, labels))


def convert_path_to_exp(path):
    exp = ""
    for i in range(len(path)):
        nd_idx = path[i]
        if nd_idx == start_state or nd_idx == end_state:
            continue

        node_content = idx_to_node[nd_idx]
        node_dic = ast.literal_eval(str(node_content))
        text = ""
        for key, value in node_dic.items():
            text = value[1]
            break

        # if i == 0 or i == len(path) - 1 :
        #     exp += ' (' + text + ')'
        # else:
        #     exp += ' ' + text
        exp += ' ' + text
    return exp

    

def merge_node(fsa, sent_id1, wd_id1, sent_id2, wd_id2):
    """
    Merge two nodes represented by (sent_id1, wd_id1, wd) and (sent_id2, wd_id2, wd) in fsa.
    """
    # Find the node of (sent_id1, wd_id1)
    # print("Sent 1: ", sent_id1)
    # print("Word 1: ", wd_id1)
    node1 = find_node(fsa, sent_id1, wd_id1)
    if node1 is None:
        return fsa
                
    # Find the node of (sent_id2, wd_id2, wd)
    # print("Sent 2: ", sent_id2)
    # print("Word 2: ", wd_id2)
    node2 = find_node(fsa, sent_id2, wd_id2)
    if node2 is None:
        return fsa

    if node1 == node2:
        return fsa

    # Create the new label
    node1_content = idx_to_node.get(node1)
    if node1_content is None:
        print("Node 1 not found during merging!")
        find_nodes_of_sent(node1)
        return fsa
    node_label1 = str(node1_content)
    node_dic1 = ast.literal_eval(node_label1)

    node2_content = idx_to_node.get(node2)
    if node2_content is None:
        print("Node 2 not found during merging!")
        find_nodes_of_sent(sent_id2)
        return fsa
    node_label2 = str(node2_content)
    node_dic2 = ast.literal_eval(node_label2)
    
    merge_label = str(merge_dics(node_dic1, node_dic2))
    # if (sent_id1 == 3 and wd_id1 == 0) or \
    #    (sent_id2 == 3 and wd_id2 == 0):
    #     print("The new label:{}".format(merge_label))        


    # Relabel node1 and remove node2
    idx_to_node[node1] = merge_label
    del idx_to_node[node2]
    #idx_to_node[node2] = ""
    
    # Merge two nodes
    #print("Merge node {} and node {}\n".format(node1, node2))
    new_fsa = nx.contracted_nodes(fsa, node1, node2)

    # if (sent_id1 == 3 and wd_id1 == 0) or \
    #    (sent_id2 == 3 and wd_id2 == 0):
    #     print_node_contents()
    return new_fsa

def gene_sents_node(fsa, cur_node, update_sents, node_sent_dic):
    """
    Generate sentences for a node based on the last node.

    TODO: remove the last_node when all its neighbors have been processed.
    """
    new_sents = []
    if cur_node == end_state:
        new_sents = update_sents
    else:
        cur_content = idx_to_node.get(cur_node)
        if cur_content is None:
            print("The node cur_node is not found!")
            return

        #print("cur_node:{}".format(cur_node))        
        #print("cur_content:{}".format(cur_content))
        cur_words = ast.literal_eval(cur_content)
        #print("cur_words:{}".format(cur_words))
        new_word = None
        for key in cur_words:
            new_word = cur_words[key][1]
            break

        #print("new word: {}".format(new_word))
        if len(update_sents) == 0:
            new_sents = [new_word]
        else:
            new_sents = [(sent + ' ' + new_word) for sent in update_sents]
    
    exist_sents = node_sent_dic.get(str(cur_node))
    all_sents = []
    if exist_sents is None:
        all_sents = new_sents
    else:
        all_sents = exist_sents + new_sents

    #print("Sents at node {}:\n {}".format(str(cur_node), all_sents))
    node_sent_dic[str(cur_node)] = all_sents

    for nd in fsa.neighbors(cur_node):
        gene_sents_node(fsa, nd, new_sents, node_sent_dic)

def generate_sents(fsa):
    """
    Generate all possible sentences represented by an fsa.
    """
    # Find the starting node
    node_sent_dic = {} # {node: list of sentences}
    start_node = start_state
    end_node = end_state
    node_sent_dic[str(start_node)] = []
    
    for nd in fsa.neighbors(start_node):
        gene_sents_node(fsa, nd, [], node_sent_dic)

    all_sents = node_sent_dic.get(str(end_node))
    # print("All sents: {}".format(all_sents))
    return all_sents

def gene_pairwise_paths_node(fsa, prev_nd, nd, pair_to_paths):
    """
    Generate paths for (nd', nd) if (nd', prev_nd) is in pair_to_paths
    """
    #print("prev_nd: {}, nd: {}".format(prev_nd, nd))
    #print("pair_to_paths: {}".format(pair_to_paths))    
    pair_to_paths_nd = {}
    for nd_pair, prev_paths in pair_to_paths.items():
        nd1, nd2 = nd_pair
        if nd2 != prev_nd:
            continue

        #print("nd_pair: {}".format(nd_pair))
        #print("prev_paths: {}".format(prev_paths))

        new_paths = []
        for path in prev_paths:
            #new_p = path.copy()
            new_p = list(path)
            new_p.append(nd)
            new_paths.append(new_p)

        #print("new_paths: {}".format(new_paths))

        nd_paths = pair_to_paths.get((nd1, nd))
        if nd_paths is None:
            nd_paths = []
        nd_paths.extend(new_paths)

        pair_to_paths_nd[(nd1, nd)] = nd_paths

    #print("pair_to_paths_nd: {}".format(pair_to_paths_nd))
    
    for key, value in pair_to_paths_nd.items():
        #print("key: {}".format(key))
        #print("value: {}".format(value))
        unique_value = [k for k,_ in itertools.groupby(sorted(value))]
        pair_to_paths[key] = unique_value

    #print("pair_to_paths updated: {}".format(pair_to_paths))            

    for nd_next in fsa.neighbors(nd):
        gene_pairwise_paths_node(fsa, nd, nd_next, pair_to_paths)
    
def generate_pairwise_paths(fsa):
    """
    Find paths between any pair of nodes
    """
    print("Generate pairwise paths")
    pair_to_paths = {} # {(node1, node2): [paths]}
    for nd in fsa:
        pair_to_paths[(nd, nd)] = [[nd]]
    
    for nd in fsa.neighbors(start_state):
        gene_pairwise_paths_node(fsa, start_state, nd, pair_to_paths)

    #print("pair_to_paths:{}".format(pair_to_paths))
    return pair_to_paths

def find_alter_path_pairs(pair_to_paths):
    """
    Find all pairwise alternative paths between pairs of nodes
    """
    pair_to_alter_paths = {}
    for pair, paths in pair_to_paths.items():
        if len(paths) < 2:
            continue
        
        disjoint_path_pairs = []
        for i in range(len(paths)):
            path_set1 = set(paths[i])
            for j in range(i+1, len(paths)):
                path_set2 = set(paths[j])
                if len(path_set1.intersection(path_set2)) == 2:
                    disjoint_path_pairs.append((paths[i], paths[j]))

        pair_to_alter_paths[pair] = disjoint_path_pairs

    return pair_to_alter_paths

def find_optional_exps(pair_to_paths):
    """
    Find all optional paths between each pair of nodes
    """
    ndpair_to_oppaths = {}
    ndpair_to_exps = {}
    for pair, paths in pair_to_paths.items():
        if pair[0] == pair[1]:
            continue
        
        has_direct_link = False
        for p in paths:
            if len(p) == 2:
                has_direct_link = True
                break

        if has_direct_link and len(paths) > 1:
            #print("Optional paths between nd {} and nd {}".format(pair[0], pair[1]))
            op_paths = []
            op_exp = []
            for i, p in enumerate(paths):
                if len(p) == 2:
                    continue

                op_paths.append(p)
                exp = convert_path_to_exp(p)
                op_exp.append(exp)
                #print("Optional expressions: {}".format(exp))
                #print("Path {}: {}".format(i, p))
            ndpair_to_exps[(pair[0], pair[1])] = op_exp
            ndpair_to_oppaths[(pair[0], pair[1])] = op_paths
        # else:
        #     print("There is no optional path between nd {} and nd {}".format(pair[0], pair[1]))
    print("ndpair_to_oppaths: {}".format(ndpair_to_oppaths))
    return ndpair_to_exps

def find_earliest_ancester(fsa, nd_set):
    for nd1 in nd_set:
        is_ancester = True        
        for nd2 in nd_set:
            if nd1 == nd2:
                continue

            if nd1 in descendants(fsa, nd2):
                is_ancester = False
                break

        if is_ancester:
            return nd1

    return None # there is a loop

def find_next_sent_node(fsa, sent_id, node):
    """
    Find the next node of the current node for sent_id
    """
    #print("Find the next node for node {}...".format(node))
    
    if node == end_state:
        return None

    reach_end = False
    next_cands = []    
    for nd_idx in fsa.neighbors(node):
        if nd_idx == end_state:
            reach_end = True
            continue
        
        node_content = idx_to_node[nd_idx]
        node_dic = ast.literal_eval(str(node_content))

        for key in node_dic:
            if key == sent_id:
                # print("Key: {}".format(key))
                # print("nd_idx: {}".format(nd_idx))
                next_cands.append(nd_idx)
                
    if len(next_cands) == 1:
        return next_cands[0]
    elif len(next_cands) > 1:
        return find_earliest_ancester(fsa, next_cands)

    if reach_end:
        return end_state
    else:
        return None

def find_rest_sent_nodes(fsa, sent_id, node):
    """
    Find all the rest of the sentence nodes of sent_id
    """
    #print("Find the rest sent nodes starting from node {} for sent {}...".format(node, sent_id))
    sent_nodes = [node]
    #print("sent_nodes: {}".format(sent_nodes))    
    next_node = find_next_sent_node(fsa, sent_id, node)
    #print("next node: {}".format(next_node))
    all_nodes = []
    if next_node is None:
        all_nodes = sent_nodes
    else:
        rest_nodes = find_rest_sent_nodes(fsa, sent_id, next_node)
        #print("The rest of the nodes starting from {}: {}".format(next_node, rest_nodes))
        all_nodes = sent_nodes + rest_nodes

    #print("All nodes: {}".format(all_nodes))
    return all_nodes


def find_sent_optional_subseq(fsa, sent_id):
    """
    Find optional expressions of a given sent_id
    """
    # Find the path of the given sentence
    start_node = start_state
    sent_path = find_rest_sent_nodes(fsa, sent_id, start_node)

    # Find all nodes that are optional
    optional_sub_paths = []
    for i, nd_idx1 in enumerate(sent_path):
        if i == len(sent_path) - 2:
            break
        
        for j, nd_idx2 in enumerate(sent_path[i+2:]):
            if nd_idx2 in fsa.successors(nd_idx1):
                optional_sub_paths.append((sent_path[i+1], sent_path[j+i+1]))

    #print("All optional paths: {}".format(optional_sub_paths))

    # Traverse the sentence path, and highlight optional nodes with parentheses
    display_sent = ""
    for nd_idx in sent_path:
        if nd_idx == start_state or nd_idx == end_state:
            continue

        node_content = idx_to_node.get(int(nd_idx))
        nd_dic = ast.literal_eval(str(node_content))
        node_word = nd_dic.get(sent_id)[1]
        begin_opt_path = [ele for ele in optional_sub_paths if ele[0] == nd_idx]
        left_par_num = len(begin_opt_path) # Number of left parentheses
        end_opt_path = [ele for ele in optional_sub_paths if ele[1] == nd_idx]
        right_par_num = len(end_opt_path) # Number of right parentheses
        
        # left_paren = ''.join("(" * left_par_num)
        # right_paren = ''.join(")" * right_par_num)
        left_paren = ''.join("[" * left_par_num)
        right_paren = ''.join("]" * right_par_num)

        
        # print("display_sent: {}".format(display_sent))
        # print("The word of the node: {}".format(node_word))
        # print("left_paren: {}".format(left_paren))
        # print("right_paren: {}".format(right_paren))
        display_sent = display_sent + left_paren + node_word + right_paren + " "
    print("The final sentence with highlighted alternative expressions: {}".format(display_sent))
    
    return display_sent


def find_verb_paraphrases(fsa, origin_sents, para_bound=3):
    """
    Find all verb paraphrases whose length do not exceed para_bound
    """
    # For each pair of nodes (nd1, nd2), find all paths from nd1 to nd2 with length not larger than para_bound+1
    nd_pair_to_paras = {}
    for nd1 in fsa:
        for nd2 in fsa:
            paths = nx.all_simple_paths(fsa, source=nd1, target=nd2, cutoff=para_bound+1)
            paths = [p for p in paths if len(p) > 2] # Remove direct links
            #print("All paths for phrase paraphrases: {}".format(paths))

            if len(paths) <= 1:
                continue

            # Remove node pairs if their alternative paths can be shrunk.
            # Check the number of distinguished neighbors of nd1 in all paths.
            nd1_neis = [p[1] for p in paths]
            if len(set(nd1_neis)) == 1:
                continue

            # Check neighbors of nd2
            nd2_neis = [p[-2] for p in paths]
            if len(set(nd2_neis)) == 1:
                continue

            paths = [p[1:-1] for p in paths]
            word_paras = []
            #print("Alternative paths: {}".format([p for p in paths]))
            #print("Alternative expressions: {}".format([convert_path_to_repr_exp(p) for p in paths]))
            for p in paths:
                # Find the corresponding sentence
                sent_ids = find_path_sent(p)
                #print("sent_ids for path {}: {}".format(p, sent_ids))
                # Find sents that has verbs
                for id in sent_ids:
                    tokens, tk_ids = convert_path_to_sent_tks(p, id)
                    exp = " ".join(tokens)
                    #print("Exp: {}".format(exp))
                    exp_doc = nlp(exp)
                    tk_count = 0
                    for tk in exp_doc:
                        if tk.pos_ != "VERB":
                            tk_count += 1
                            continue

                        sent = origin_sents[id]
                        sent_doc = nlp(sent)
                        vp = find_dobj_phrase(sent_doc, tk_ids[tk_count])
                        if vp is not None:
                            word_paras.append(vp)

                        vpp = find_prep_pobj_phrase(sent_doc, tk_ids[tk_count])
                        if vpp is not None:
                            word_paras.append(vpp)

            word_paras = list(set(word_paras))
            #print("Candidate paraphrases: {}".format(word_paras))
            if len(word_paras) > 1:
                nd_pair_to_paras[(nd1, nd2)] = word_paras
            else:
                nd_pair_to_paras[(nd1, nd2)] = []

    #print("All word paraphrases of maximum length {}:\n {}".format(para_bound, nd_pair_to_paras))
    return nd_pair_to_paras

def old_find_verb_paraphrases(fsa, origin_sents, para_bound=3):
    """
    Find all verb paraphrases whose length do not exceed para_bound
    """
    # For each pair of nodes (nd1, nd2), find all paths from nd1 to nd2 with length not larger than para_bound+1
    nd_pair_to_paras = {}
    for nd1 in fsa:
        for nd2 in fsa:
            paths = nx.all_simple_paths(fsa, source=nd1, target=nd2, cutoff=para_bound+1)
            paths = [p for p in paths if len(p) > 2] # Remove direct links
            #print("All paths for phrase paraphrases: {}".format(paths))

            if len(paths) <= 1:
                continue
            
            # Remove node pairs if their alternative paths can be shrunk.
            # Check the number of distinguished neighbors of nd1 in all paths.
            nd1_neis = [p[1] for p in paths]
            if len(set(nd1_neis)) == 1:
                continue

            # Check neighbors of nd2
            nd2_neis = [p[-2] for p in paths]
            if len(set(nd2_neis)) == 1:
                continue
            
            word_paras = []
            #print("Alternative paths: {}".format([convert_path_to_exp(p) for p in paths]))
            for p in paths:
                # Find the corresponding sentence
                sent_ids = find_path_sent(p)
                # Find sents that has verbs
                for id in sent_ids:
                    tokens, tk_ids = convert_path_to_sent_tks()
                    exp = " ".join(tokens)

                # Find the target verb in each sentence
                for nd in p:
                    nd_content = idx_to_node[nd]
                    #print("nd_content: {}".format(nd_content))
                    if nd_content == str(start_state) or nd_content == str(end_state):
                        continue
                    
                    node_dic = ast.literal_eval(str(nd_content))
                    for sent_id, tk in node_dic.items():
                        tk_idx = tk[0]
                        tk_text = tk[1]
                        tk_doc = nlp(tk_text)
                        if tk_doc[0].pos_ != "VERB":
                            continue
                        # if tk_doc[0].pos_ != "VERB" and tk_doc[0].text != "Book":
                        #     continue
                        sent = origin_sents[sent_id]
                        sent_doc = nlp(sent)
                        vp = find_dobj_phrase(sent_doc, tk_idx)
                        word_paras.append(vp)

            word_paras = list(set(word_paras))
            if len(word_paras) > 1:
                nd_pair_to_paras[(nd1, nd2)] = word_paras
            else:
                nd_pair_to_paras[(nd1, nd2)] = []

    #print("All word paraphrases of maximum length {}:\n {}".format(para_bound, nd_pair_to_paras))
    return nd_pair_to_paras



def find_phrase_paraphrases(fsa, para_bound=3):
    """
    Find all paraphrases whose length do not exceed para_bound
    """
    # For each pair of nodes (nd1, nd2), find all paths from nd1 to nd2 with length not larger than para_bound+1
    nd_pair_to_paras = {}
    for nd1 in fsa:
        for nd2 in fsa:
            paths = nx.all_simple_paths(fsa, source=nd1, target=nd2, cutoff=para_bound+1)
            paths = [p for p in paths if len(p) > 2] # Remove direct links
            #print("All paths for phrase paraphrases: {}".format(paths))

            if len(paths) <= 1:
                continue
            
            # Remove node pairs if their alternative paths can be shrunk.
            # Check the number of distinguished neighbors of nd1 in all paths.
            nd1_neis = [p[1] for p in paths]
            if len(set(nd1_neis)) == 1:
                continue

            # Check neighbors of nd2
            nd2_neis = [p[-2] for p in paths]
            if len(set(nd2_neis)) == 1:
                continue
            
            word_paras = []
            for p in paths:
                assert(len(p) > 2)
                exp = convert_path_to_exp(p[1:-1]) # Do not include the start node and end node
                word_paras.append(exp)

            nd_pair_to_paras[(nd1, nd2)] = word_paras

    #print("All word paraphrases of maximum length {}:\n {}".format(para_bound, nd_pair_to_paras))
    return nd_pair_to_paras

def find_path_sent(path):
    """
    Find all sentences covered by the path
    """
    sent_ids = []
    cand_sent_ids = []
    for nd in path:
        if nd == start_state or nd == end_state:
            continue
            
        node_content = idx_to_node[nd]
        node_dic = ast.literal_eval(str(node_content))
        nd_sent_ids = node_dic.keys()
        #print("node content for nd {}: {}".format(nd, node_dic))
        cand_sent_ids.append(nd_sent_ids)

    assert(len(cand_sent_ids) > 0)
    sent_ids = cand_sent_ids[0]
    for id_set in cand_sent_ids:
        sent_ids = set(sent_ids).intersection(set(id_set))

    return sent_ids

def convert_path_to_sent_tks(path, sent_idx):
    """
    Convert a path of sent_idx into an expression
    """
    tokens = []
    tk_ids = []
    for i in range(len(path)):
        nd_idx = path[i]
        if nd_idx == start_state or nd_idx == end_state:
            continue

        node_content = idx_to_node[nd_idx]
        node_dic = ast.literal_eval(str(node_content))
        tk_info = node_dic.get(sent_idx)
        if tk_info is None:
            print("Warning: cannot convert path {} for sent_idx {}".format(path, sent_idx))
        tk_ids.append(tk_info[0])
        tokens.append(tk_info[1])

    return tokens, tk_ids

def convert_path_to_repr_exp(path, with_end=False):
    """
    Generate a representative expression for the given path
    """
    exp = ""
    #print("Path: {}".format(path))    
    for i in range(len(path)):
        if with_end == False and \
           ((i == 0) or (i == len(path)-1)):
            continue
        
        nd_idx = path[i]
        if nd_idx == start_state:
            exp += "BOS"
            continue

        if nd_idx == end_state:
            exp += "EOS"
            continue
        
        node_content = idx_to_node[nd_idx]
        #print("Node content: {}".format(node_content))
        node_dic = ast.literal_eval(str(node_content))
        text = ""
        for key, value in node_dic.items():
            text = value[1]
            break

        exp += ' ' + text
    return exp

def print_alter_paths(pair_to_alter_paths):
    print("Alternative expressions\n")
    for _, value in pair_to_alter_paths.items():
        if len(value) == 0:
            continue

        for path_pair in value:
            path1, path2 = path_pair
            exp1 = convert_path_to_exp(path1)
            exp2 = convert_path_to_exp(path2)
            print(exp1)
            print(exp2)
            print("\n")

        print("\n\n\n")

def print_node_contents():
    print("node labels: {}\n".format(idx_to_node))


def add_sent_fsa(fsa, sent, sent_id):
    """
    Add a sent into the fsa by only merging the start and end nodes.
    """
    global globe_node_idx    

    fsa.add_node(start_state)
    prev_node = start_state
    token_idx = 0
    for i in range(len(sent)):
        node_text = str({sent_id: [i, sent[i]]})
        fsa.add_edge(prev_node, globe_node_idx)
        idx_to_node[globe_node_idx] = node_text
        node_to_idx[node_text] = globe_node_idx
        prev_node = globe_node_idx
        globe_node_idx += 1
        token_idx += 1
    
    fsa.add_edge(prev_node, end_state)
    return fsa

def get_node_text(fsa, nd_id):
    node_content = idx_to_node.get(int(nd_id))
    return node_content

def get_repr_nd_text(fsa, nd_id):
    """
    Obtain the representative node contents
    """
    node_content = idx_to_node.get(int(nd_id))
    #print("Node content: {}".format(node_content))
    if node_content == str(start_state):
        return str(start_state)

    if node_content == str(end_state):
        return str(end_state)
    
    node_ctt_dict = ast.literal_eval(str(node_content))
    node_repr_str = node_ctt_dict.items()[1][1][1]

    return node_repr_str

def qualify_for_merge(sents_dic, align_matrix, sent_id1, sent_id2):
    """
    Check whether two sentences sent_id1 and sent_id2 need to be merged.
    """
    # if sent_id2 >= sent_id1:
    #     tmp = sent_id2
    #     sent_id2 = sent_id1
    #     sent_id1 = tmp
    
    # align_list = find_align_element(align_matrix, sent_id1, sent_id2)
    # sent1 = sents_dic[sent_id1]
    # sent2 = sents_dic[sent_id2]
    # has_noun = False
    # for align in align_list:
    #     phrase = sent1[align[0]-1]
    #     doc = nlp(phrase)
    #     for tk in doc:
    #         if tk.pos_ == 'NOUN':
    #             return True
    # return False
    return True


def process_sents(fsa, sents_dic, align_matrix, sent_ids):
    """
    Combine aligned nodes of sentences in the same cluster

    parameters:
    sents: {id: tokenized sent list}
    align_matrix: {(sent_id1,sent_id2): alignment list}
    """
    for sent_id1 in sent_ids:
        for sent_id2 in sent_ids:
            if sent_id2 >= sent_id1:
                continue

            #print("Handling sent {} (length {}, text {}) \n and sent {} (length {}, text {})".format(sent_id1, len(sents_dic[sent_id1]), sents_dic[sent_id1], sent_id2, len(sents_dic[sent_id2]), sents_dic[sent_id2]))
            align_list = find_align_element(align_matrix, sent_id1, sent_id2)
            if not qualify_for_merge(sents_dic, align_matrix, sent_id1, sent_id2):
                print("sent {} and sent {} do not qualify for merge.".format(sent_id1, sent_id2))
                continue # Do not align two sentences if not qualified.
        
            for pair in align_list:
                #print("Align_list: {}".format(align_list))
                align_idx1 = pair[0]
                align_idx2 = pair[1]

                wd_id1 = pair[0] - 1
                wd_id2 = pair[1] - 1
                fsa = merge_node(fsa, sent_id1, wd_id1, sent_id2, wd_id2)
                    
    # Check cycles
    try:
        cycles = list(nx.find_cycle(fsa))
        print("Cycles found: {}".format(cycles))
        nodes_in_cycle = []
        for nd_pair in cycles:
            n1, n2 = nd_pair
            if n1 not in nodes_in_cycle:
                nodes_in_cycle.append(n1)
                print("Node {}: {}".format(n1, idx_to_node[n1]))

            if n2 not in nodes_in_cycle:
                nodes_in_cycle.append(n2)
                print("Node {}: {}".format(n2, idx_to_node[n2]))
    except nx.exception.NetworkXNoCycle:
        pass


    return fsa

def display_graph(fsa):
    print("Displaying graph contents and structure...\n")
    for nd_idx, content in idx_to_node.items():
        print("Node {}: \n".format(nd_idx))
        print("Contents: {} \n".format(content))
        print("Next hops: ")
        next_indices = []
        for nd in fsa.neighbors(nd_idx):
            next_indices.append(nd)
        print("{} \n".format(next_indices))
    
    
