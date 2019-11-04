import sys
sys.path.append('./scripts/essentia')

import json
import flask
import networkx as nx
from networkx.readwrite import json_graph
# from scripts.fsa import *
# from scripts.generate_word_alignment import *
# from scripts.preprocessing import create_valid_groups
from generate_word_alignment import nlp, make_alignment_matrix, make_alignment_matrix_with_rules
from preprocessing import create_valid_groups
from fsa import create_fsa, process_sents, generate_pairwise_paths, find_phrase_paraphrases, idx_to_node, start_state, end_state

sultan_aligner = True

def prep_graph(G_temp, names):
    # Adding the names to the graph
    print("names: {}".format(names))
    for n, n_data in G_temp.nodes(data=True):
        n_data['name'] = names[n]
    # Collapsing long paths
    collapse_paths(G_temp)
    # The graph needs attribute name (used when mouse hover over nodes) and
    # a weight attribute on each edge
    G = nx.MultiDiGraph()
    node2id, id2node = {}, {}
    for id_, n in enumerate(G_temp):
        node2id[n] = id_
        id2node[id_] = n
    for i in range(len(id2node)):
        if id2node[i] == 0 or id2node[i] == 1:
            x_pos = 100 + id2node[i] * 700
            G.add_node(i, name=G_temp.nodes[id2node[i]]['name'],
                       group=id2node[i], size=8, fixed=True, x=x_pos, y=200)
        else:
            G.add_node(i, name=G_temp.nodes[id2node[i]]['name'],
                       group=2, size=5, fixed=False)
    for (x, y) in G_temp.edges():
        G.add_edge(node2id[x], node2id[y], weight=1)
    return G


def collapse_paths(G):
    has_collapse = True
    while has_collapse:
        has_collapse = False
        for (x, y) in G.edges():
            if G.in_degree(x) == 1 and G.out_degree(x) == 1 and \
                    G.in_degree(y) == 1 and G.out_degree(y) == 1:
                has_collapse = True
                new_node = str(x) + ' ' + str(y)
                new_name = G.nodes[x]['name'] + ' ' + G.nodes[y]['name']
                G.add_node(new_node, name=new_name)
                for (z, _) in G.in_edges(x):
                    G.add_edge(z, new_node)
                for (_, z) in G.out_edges(y):
                    G.add_edge(new_node, z)
                G.remove_nodes_from([x, y])
                break
    return G


def build_graph_test(sents):
    # TODO: We should make a graph from sentences
    # This is a dummy solution for now.
    G = nx.read_adjlist('example.adjlist', create_using=nx.MultiDiGraph(),
                        nodetype=int)
    raw_names = json.load(open('node_to_text_dic.json', 'r'))
    names = {}
    for node_str, values_str in raw_names.items():
        node = int(node_str)
        if node == 0:
            names[node] = 'START'
        elif node == 1:
            names[node] = 'END'
        else:
            values = eval(values_str) if values_str != "" else {}
            all_words = list(set([values[x][1].lower() for x in values]))
            names[node] = '/'.join(all_words)
    return G, names


def build_graph(sents):
    origin_sents = sents
    tk_sents = {}
    for i, sent in enumerate(sents):
        doc = nlp(sent)
        tk_st = [tk.text for tk in doc]
        tk_sents[i] = tk_st

    align_matrix = None
    sents_cluster = None
    if sultan_aligner:
        align_matrix = make_alignment_matrix(origin_sents)
        #merge_chunks(align_matrix, tk_sents, origin_sents)
        sents_cluster = create_valid_groups(align_matrix, tk_sents)
    else:
        align_matrix = make_alignment_matrix_with_rules(origin_sents)
        sents_cluster = create_valid_groups(align_matrix, tk_sents)
        #sents_cluster = [range(len(align_matrix))]
    # print("sentence clusters: {}".format(sents_cluster))
    # print(align_matrix)

    fsa = create_fsa(tk_sents)
    for cluster in sents_cluster:
        fsa = process_sents(fsa, tk_sents, align_matrix, cluster)

    raw_names = idx_to_node
    names = {}
    for node_str, values_str in raw_names.items():
        # print("node_str: {}".format(node_str))
        # print("values_str: {}".format(values_str))        
        node = int(node_str)
        if node == start_state:
            names[node] = 'START'
        elif node == end_state:
            names[node] = 'END'
        else:
            values = eval(values_str) if values_str != "" else {}
            all_words = list(set([values[x][1].lower() for x in values]))
            names[node] = '/'.join(all_words)
    return fsa, names


def main():
    print('Wrote node-link JSON data to force/force.json')

    # Serve the file over http to allow for cross origin requests
    app = flask.Flask(__name__, static_folder="force")

    @app.route('/<path:path>')
    def static_proxy(path):
        return flask.send_from_directory(app.static_folder, path)

    @app.route('/')
    def index():
        return flask.send_from_directory(app.static_folder, "index.html")

    @app.route('/render', methods=['POST'])
    def renderit():
        sents = flask.request.form['sents'].split('\r\n')
        G, names = build_graph(sents)
        # writing optional expressions
        #pair_to_paths = generate_pairwise_paths(G)
        #print("pair_to_paths: {}".format(pair_to_paths))
        #ndpair_to_exps = find_optional_exps(pair_to_paths)
        #print("ndpair_to_exps: {}".format(ndpair_to_exps))

        # Output alternative expressions
        nd_pair_to_paras = find_phrase_paraphrases(G)
        
        with open('./demo/force/alt_exp.txt', 'w') as out_file:
            #out_file.write("Optional expressions:\n")
            # out_file.write('This is a test!\n')
            # out_file.write('    (1) This is another test!\n')
            # out_file.write('    (*) This is nothing!\n')
            count = 0
            for _, v in nd_pair_to_paras.items():
                # for exp in v:
                #     out_file.write(exp)
                #     out_file.write('\n')
                #print("v: {}".format(v))
                out_file.write("Group {}: ".format(count))
                out_file.write(str(v))
                out_file.write('\n\n')
                count += 1

        # Post-processing for demo
        G = prep_graph(G, names) # merge consecutive nodes together for demo
        # write json formatted data
        d = json_graph.node_link_data(G)  # node-link format to serialize
        # write json
        json.dump(d, open('./demo/force/force.json', 'w'))

        return flask.send_from_directory(app.static_folder, "force.html")

    # this is to avoid caching
    @app.after_request
    def add_header(r):
        """
        Add headers to both force latest IE rendering engine or Chrome Frame,
        and also to cache the rendered page for 10 minutes.
        """
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        r.headers['Cache-Control'] = 'public, max-age=0'
        return r

    app.run(port=8000)


if __name__ == "__main__":
    main()
