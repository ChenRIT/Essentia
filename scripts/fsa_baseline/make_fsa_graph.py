import sys
import spacy
import networkx as nx
from collections import defaultdict
from benepar.spacy_plugin import BeneparComponent
from tqdm import tqdm


class Node:

    def __init__(self, tag, span):
        # ignore punctuation
        self.tag = tag
        self.span = span
        self.rule = None
        self.children = []

    def add_child(self, node):
        if node.tag in ['.', ',', '?', '!']:
            return
        self.children.append(node)

    def make_rule(self):
        self.rule = ' '.join(x.tag for x in self.children)

    def make_forest_node(self):
        forest_node = ForestNode(self.tag)
        words = set()
        all_words = set()
        for t in self.span:
            if t.pos_ == 'NOUN' or t.pos_ == 'ADJ' or \
                    t.pos_ == 'PROPN' or t.pos_ == 'VERB' or \
                    t.pos_ == 'ADP':
                words.add(str(t))
            all_words.add(str(t))
        forest_node.words = words
        forest_node.all_words = all_words
        # one possible application of the rule
        if self.rule != '':
            forest_node.children[self.rule].append([])
        for child_node in self.children:
            forest_node.children[self.rule][0].append(
                child_node.make_forest_node())
        return forest_node

    def __str__(self):
        inside = ' '.join([self.tag] + [str(x) for x in self.children])
        return '(' + inside + ')'

    __repr__ = __str__


class ForestNode:

    def __init__(self, tag):
        # ignore punctuation
        self.tag = tag
        self.words = set()
        self.all_words = set()
        self.children = defaultdict(list)

    def merge_tree(self, root):
        assert(root.tag == self.tag)
        # adding the words together
        self.words.update(root.words)
        self.all_words.update(root.all_words)
        # updating the children
        for rule, nodes in root.children.items():
            # if the rule exists (further merge is required)
            if rule in self.children:
                # we need to check with each possible set of children
                # and merge only if compatible
                any_compatible = False
                for childlist in self.children[rule]:
                    if compatible(childlist, root.children[rule][0]):
                        any_compatible = True
                        for i, child in enumerate(childlist):
                            child.merge_tree(root.children[rule][0][i])
                if not any_compatible:
                    self.children[rule].append(root.children[rule][0])
            # if the rule does not exist
            else:
                self.children[rule] = root.children[rule]

    def __str__(self):
        rules_text_list = [self.tag + ' --> ' + rule + '\n' +
                           '\n-----\n'.join(['\n'.join([str(x) for x in alist])
                                             for alist in nodes_lists])
                           for rule, nodes_lists in self.children.items()]
        indented_list = [x.replace('\n', '\n   ') for x in rules_text_list]
        if len(indented_list) == 0:
            inside = self.tag + ' ' + '+'.join(self.all_words)
        else:
            inside = '\n'.join([self.tag] + indented_list)
        return inside

    __repr__ = __str__


def compatible(childrenA, childrenB):
    # the words should not appear in other blocks
    assert len(childrenA) == len(childrenB)
    for i, childA in enumerate(childrenA):
        for j, childB in enumerate(childrenB):
            if i == j:
                continue
            if len(set(childA.words).intersection(set(childB.words))) > 0:
                return False
    return True


def load_file(filepath):
    with open(filepath) as ofile:
        return [x.rstrip() for x in ofile.readlines()]


def span_to_tree(span, ind=0):
    children = list(span._.children)
    labels = list(span._.labels)
    num_labels = len(labels) - ind
    if num_labels > 1:
        tag = span._.labels[ind]
        node = Node(tag, span)
        node.add_child(span_to_tree(span, ind=ind+1))
    else:
        tag = span[ind].tag_ if num_labels == 0 else span._.labels[ind]
        node = Node(tag, span)
        if len(children) == 0 and num_labels > 0:
            # need to add the tag as well
            leaf_node = Node(span[ind].tag_, span)
            leaf_node.make_rule()
            node.add_child(leaf_node)
        else:
            for x in children:
                node.add_child(span_to_tree(x))
    node.make_rule()
    return node


def make_parse_trees(sents):
    nlp = spacy.load('en')
    nlp.add_pipe(BeneparComponent("benepar_en2"))
    tree_list = []
    for s in tqdm(sents):
        span = list(nlp(unicode(s, "utf-8")).sents)[0]
        root_node = Node('ROOT', span)
        node = span_to_tree(span)
        root_node.add_child(node)
        root_node.make_rule()
        tree_list.append(root_node)
    return tree_list


def make_forest(trees):
    forest = ForestNode('ROOT')
    for t in trees:
        forest.merge_tree(t.make_forest_node())
    return forest


def node_to_path(G, start_node, end_node, forest, used_ids):
    if len(forest.children) == 0:
        # no recursion
        for w in forest.all_words:
            G.add_edge(start_node, end_node, key=w, attr_dict={'text': w})
    else:
        # recursion
        for rule, nodes_lists in forest.children.items():
            for nodes in nodes_lists:
                start_child = start_node
                for i, node in enumerate(nodes):
                    if i + 1 == len(nodes):
                        end_child = end_node
                    else:
                        end_child = used_ids[-1] + 1
                        used_ids.append(end_child)
                        G.add_node(end_child)
                    node_to_path(G, start_child, end_child, node, used_ids)
                    start_child = end_child


def make_fsa(forest):
    G = nx.MultiDiGraph()
    G.add_node(0)
    G.add_node(1)
    start_node = 0
    end_node = 1
    used_ids = [0, 1]
    node_to_path(G, start_node, end_node, forest, used_ids)
    return G


def clean_up_keys(G):
    H = G.copy()
    for u, v, k, dat in H.edges(keys=True, data=True):
        if type(k) != str:
            G.remove_edge(u, v, k)
            G.add_edge(u, v, key=dat['attr_dict']['text'],
                       attr_dict=dat['attr_dict'])


def squeeze_in_edges(fsa, node_to_order):
    # checking each node
    for n in fsa.nodes():
        # checking in edges
        ins = fsa.in_edges(n, data=True)
        word_to_node = defaultdict(set)
        for (x, _, data) in ins:
            if data['attr_dict']['text'] != '':
                word_to_node[data['attr_dict']['text']].add(x)
        for word, to_merge in word_to_node.items():
            if len(to_merge) <= 1:
                continue
            a = list(to_merge)[0]
            b = list(to_merge)[1]
            if node_to_order[a] > node_to_order[b]:
                (a, b) = (b, a)
            if nx.has_path(fsa, a, b):
                # split is required
                new_node = max([k for k in node_to_order]) + 1
                fsa.add_node(new_node)
                fsa.add_edge(a, new_node, key='', attr_dict={'text': ''})
                fsa.remove_edge(a, n, key=word)
                fsa.add_edge(new_node, n, key=word, attr_dict={'text': word})
                fsa = nx.contracted_nodes(fsa, b, new_node)
                clean_up_keys(fsa)
            else:
                # removing the duplicate edge
                fsa.remove_edge(a, n, key=word)
                fsa = nx.contracted_nodes(fsa, a, b)
                clean_up_keys(fsa)
            return True, fsa
    return False, fsa


def squeeze_out_edges(fsa, node_to_order):
    # checking each node
    for n in fsa.nodes():
        # checking in edges
        outs = fsa.out_edges(n, data=True)
        word_to_node = defaultdict(set)
        for (_, x, data) in outs:
            if data['attr_dict']['text'] != '':
                word_to_node[data['attr_dict']['text']].add(x)
        for word, to_merge in word_to_node.items():
            if len(to_merge) <= 1:
                continue
            a = list(to_merge)[0]
            b = list(to_merge)[1]
            if node_to_order[a] > node_to_order[b]:
                (a, b) = (b, a)
            if nx.has_path(fsa, a, b):
                # split is required
                new_node = max([k for k in node_to_order]) + 1
                fsa.add_node(new_node)
                fsa.add_edge(new_node, b, key='', attr_dict={'text': ''})
                fsa.remove_edge(n, b, key=word)
                fsa.add_edge(n, new_node, key=word, attr_dict={'text': word})
                fsa = nx.contracted_nodes(fsa, a, new_node)
                clean_up_keys(fsa)
            else:
                # split is required
                # removing the duplicate edge
                fsa.remove_edge(n, b, key=word)
                fsa = nx.contracted_nodes(fsa, a, b)
                clean_up_keys(fsa)
            return True, fsa
    return False, fsa


def squeeze_fsa(fsa):
    squeeze_possible = True
    while squeeze_possible:
        # fix keys if broken
        for (u, v, k, dat) in fsa.edges(data=True, keys=True):
            word = dat['attr_dict']['text']
            if k != word:
                fsa.remove_edge(u, v, k)
                fsa.add_edge(u, v, key=word, attr_dict={'text': word})
        # starting the process
        topological_order = list(nx.topological_sort(fsa))
        node_to_order = {k: i for i, k in enumerate(topological_order)}
        squeeze_possible, fsa = squeeze_in_edges(fsa, node_to_order)
        if squeeze_possible:
            continue
        squeeze_possible, fsa = squeeze_out_edges(fsa, node_to_order)
    return fsa


def generate_paraphrases(G, start=0):
    if start == 1:
        return ['']
    if 'paras' in G.nodes[start]:
        return G.nodes[start]['paras']
    all_sents = []
    for (_, hop, dat) in G.out_edges(start, data=True):
        word = dat['attr_dict']['text']
        hop_sents = generate_paraphrases(G, start=hop)
        all_sents += [word + ' ' + x for x in hop_sents]
    G.nodes[start]['paras'] = all_sents
    return all_sents


def move_forward(G, node, step, path, max_steps):
    if step == max_steps:
        return []
    results = []
    for (_, x, data) in G.out_edges(node, data=True):
        results.append((x, data['attr_dict']['text'], path))
        step_increment = 1 if data['attr_dict']['text'] != '' else 0
        for (y, text, p) in move_forward(G, x, step + step_increment,
                                         path + [x], max_steps):
            if text == '' or data['attr_dict']['text'] == '':
                results.append((y, data['attr_dict']['text'] + text, p))
            else:
                results.append((y, data['attr_dict']['text'] + ' ' + text, p))
    return results


def generate_alt_exps(G, max_steps=3):
    list_of_groups = []
    for n in G.nodes():
        node_results = move_forward(G, n, 0, [], max_steps)
        # merging texts with same end-node
        dst_dict = defaultdict(list)
        for (dst, text, p) in node_results:
            if text != '':
                dst_dict[dst].append((text, p))
        for k, v in dst_dict.items():
            if len(v) <= 1:
                continue
            # is it a valid path
            counts = defaultdict(int)
            for (text, p) in v:
                for n in p:
                    counts[n] += 1
            valid = True
            for _, c in counts.items():
                if c == len(v):
                    valid = False
            if not valid:
                continue
            v = [text for text, _ in v]
            list_of_groups.append(list(set(v)))
    # post processing the results
    final_list = []
    for group in list_of_groups:
        final_list.append('+++'.join(sorted(group)))
    final_list = list(sorted(set(final_list)))
    list_of_groups = list(sorted([x.split('+++') for x in final_list], key=len))
    return list_of_groups


def find_sent_nodes(G, tokens, node=0):
    if len(tokens) == 0:
        return []
    first_token = tokens[0]
    for (_, x, data) in G.out_edges(node, data=True):
        if data['attr_dict']['text'] == first_token:
            res = find_sent_nodes(G, tokens[1:], node=x)
            if res is not None:
                return [(x, data['attr_dict']['text'])] + res
        elif data['attr_dict']['text'] == '':
            res = find_sent_nodes(G, tokens, node=x)
            if res is not None:
                return [(x, data['attr_dict']['text'])] + res
    return None


def generate_opt_exps(G, G_skip, s, nlp):
    tokens = [str(t) for t in list(nlp(s).sents)[0]
              if t.tag_ not in ['.', ',', '?', '!']]
    sent_nodes = [(0, None)] + find_sent_nodes(G, tokens)
    assert sent_nodes is not None
    # iterating over pairs of nodes and see if they can be optional
    optional_start = defaultdict(int)
    optional_end = defaultdict(int)
    for i in range(len(sent_nodes) - 1):
        for j in range(i + 1, len(sent_nodes)):
            if nx.has_path(G_skip, sent_nodes[i][0], sent_nodes[j][0]):
                valid = False
                for k in range(i, j):
                    if sent_nodes[k + 1][1] != '':
                        valid = True
                if valid:
                    optional_start[i] += 1
                    optional_end[j] += 1
    # process sentence
    augmented_text = ''
    for i in range(len(sent_nodes) - 1):
        src = sent_nodes[i][0]
        dst, key = sent_nodes[i + 1]
        token = G.get_edge_data(src, dst, key=key)['attr_dict']['text']
        token = '[' * optional_start[i] + token + ']' * optional_end[i + 1]
        if token != '':
            augmented_text += token + ' '
    return augmented_text.strip()


def make_epsilon_graph(G):
    G_skip = G.copy()
    to_remove = []
    for (x, y, k, data) in G_skip.edges(keys=True, data=True):
        if data['attr_dict']['text'] != '':
            to_remove.append((x, y, k))
    for (x, y, k) in to_remove:
        G_skip.remove_edge(x, y, key=k)
    return G_skip


def main():
    sents = load_file(sys.argv[1])
    mode = sys.argv[2]
    assert mode in ['par', 'alt', 'opt']
    trees = make_parse_trees(sents)
    # printing each tree
    '''
    for i, t in enumerate(trees):
        print('(' + str(i) + ') ' + sents[i])
        print('(' + str(i) + ') ' + str(t))
        print('-' * 20)
    '''
    # merging trees
    forest = make_forest(trees)
    # making the FSA
    fsa = make_fsa(forest)
    # squeeze FSA
    G = squeeze_fsa(fsa)
    if mode == 'par':          # printing all possible paths
        new_sents = list(set([' '.join(x.lower().split())
                              for x in generate_paraphrases(G)]))
        for s in new_sents:
            print(s)
    elif mode == 'opt':
        nlp = spacy.load('en')
        G_skip = make_epsilon_graph(G)
        opt_exps = [generate_opt_exps(G, G_skip, s, nlp) for s in sents]
        for exp in opt_exps:
            print(exp)
    elif mode == 'alt':        # printing all alternative expressions
        alt_exps = generate_alt_exps(G)
        for exps in alt_exps:
            print(exps)


if __name__ == "__main__":
    main()
