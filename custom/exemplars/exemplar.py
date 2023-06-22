'''
# -*- coding: utf-8 -*-
nchen
FilePath: /rethinking_code/custom/exemplars/exemplar.py
'''

import random
from tqdm import tqdm
from tree_sitter import Language, Parser
import networkx as nx
from tree_sitter import Language, Parser
from networkx.drawing.nx_agraph import to_agraph
import pygraphviz as pgv
import numpy as np
import heapq
from itertools import count
import multiprocessing
import logging
import sys
logger = logging.getLogger(__name__)
import os
import pickle
# sys.setrecursionlimit(10000) # treat long ast
# import multiprocessing
def get_lang_by_task(task, sub_task):
    if task in ['summarize','complete']:
        return sub_task
    elif task in ['refine','generate','clone']:
        return 'java'
    elif task == 'translate':
        if sub_task == 'cs-java':
            return 'c_sharp'
        else:
            return 'java'
    elif task == 'defect':
        return 'c'
    else:
        raise 'java'

class Exemplar():
    def __init__(self, args,examples):
        self.args = args
        self.examples = examples
        self.lang =  get_lang_by_task(args.task, args.sub_task)
        LANGUAGE = Language('build/my-languages.so', self.lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        self.parser = parser
        # self.pool = multiprocessing.Pool(self.args.cpu_count)
    

    # SOY_PATH = "/path_to_your_python_binding.so"

    def get_ast_nx(self,code, parser):
        tree = parser.parse(bytes(code, 'utf-8'))
        G = nx.Graph()
        cursor = tree.walk()
        self.traverse(cursor, G, code, came_up=False, node_tag=0, node_sum=0, parent_dict={})
        return G, code

    def traverse(self,cursor, G, code, came_up, node_tag, node_sum, parent_dict):
        if not came_up:
            start = cursor.node.start_point
            end = cursor.node.end_point
            token = self.index_to_code_token(start, end, code)
            G.add_node(node_tag, label=token, features=cursor.node)
            if node_tag in parent_dict.keys():
                G.add_edge(parent_dict[node_tag], node_tag)
            if cursor.goto_first_child():
                node_sum += 1
                parent_dict[node_sum] = node_tag
                self.traverse(cursor, G, code, came_up=False, node_tag=node_sum, node_sum=node_sum, parent_dict=parent_dict)
            elif cursor.goto_next_sibling():
                node_sum += 1
                parent_dict[node_sum] = parent_dict[node_tag]
                self.traverse(cursor, G, code, came_up=False, node_tag=node_sum, node_sum=node_sum, parent_dict=parent_dict)
            elif cursor.goto_parent():
                node_tag = parent_dict[node_tag]
                self.traverse(cursor, G, code, came_up=True, node_tag=node_tag, node_sum=node_sum, parent_dict=parent_dict)
        else:
            if cursor.goto_next_sibling():
                node_sum += 1
                parent_dict[node_sum] = parent_dict[node_tag]
                self.traverse(cursor, G, code, came_up=False, node_tag=node_sum, node_sum=node_sum, parent_dict=parent_dict)
            elif cursor.goto_parent():
                node_tag = parent_dict[node_tag]
                self.traverse(cursor, G, code, came_up=True, node_tag=node_tag, node_sum=node_sum, parent_dict=parent_dict)

    def index_to_code_token(self,start_point,end_point, code):
        code = code.split('\n')
        if start_point[0] == end_point[0]:
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]
        return s

    def get_sast(self,T, leaves, tokens_dict, tokens_type_dict):
        # print("len(leaves), len(tokens_dict), len(tokens_type_dict)", len(leaves), len(tokens_dict), len(tokens_type_dict))
        
        # add subtoken edges and Data flow edges to T
        T = nx.Graph(T)
        subtoken_edges = []
        dataflow_edges = []
        identifier_dict = {}
        i = 0
        for leaf in leaves:
            token_type = tokens_type_dict[leaf]
            token = tokens_dict[leaf]
            if token_type == 'identifier':
                if token not in identifier_dict:
                    identifier_dict[token] = leaf
                else:
                    dataflow_edges.append((identifier_dict[token], leaf))
                    identifier_dict[token] = leaf
            if i > 0:
                subtoken_edges.append((old_leaf, leaf))
            old_leaf = leaf
            i += 1
        T.add_edges_from(subtoken_edges)
        T.add_edges_from(dataflow_edges)
        return T  # new_T

    def get_token_distance(self, leaves, sast, distance_metric='shortest_path_length'):  # 4min
        # print('get token distance')
        if distance_metric == 'shortest_path_length':
            ast_distance = nx.shortest_path_length(sast)
        elif distance_metric == 'simrank_similarity':
            ast_distance = nx.simrank_similarity(sast)
        # print(list(ast_distance))
        leaf=leaves
        token_num = len(leaves)
        distance = np.zeros((token_num, token_num))
        ast_distance = dict(ast_distance)
        for j in range(token_num):
            for k in range(token_num):
                if leaf[k] in ast_distance[leaf[j]].keys():
                    distance[j][k] = ast_distance[leaf[j]
                                                ][leaf[k]]  # just token distance

        return distance


    def calculate_global_efficiency(self,G, leaves):
        n = len(leaves)
        sum_inverse_distance = 0.0
        for i in range(n):
            for j in range(i+1, n):
                try:
                    d = nx.shortest_path_length(G, source=leaves[i], target=leaves[j])
                    sum_inverse_distance += 1/d
                except nx.NetworkXNoPath:
                    continue  # ignore if no path exists
        GE = sum_inverse_distance / (0.5 * n * (n - 1))
        L = 1 / GE if GE != 0 else float('inf')  # handle division by zero
        return GE, L
    def find_identifiers(self, node, code):
        if node.type == "identifier":
            start_byte = node.start_byte
            end_byte = node.end_byte
            identifier = code[start_byte:end_byte]

            if identifier in self.identifier_positions:
                self.identifier_positions[identifier].append(start_byte)
            else:
                self.identifier_positions[identifier] = [start_byte]

        for child in node.children:
            self.find_identifiers(child, code)

    def get_identifier_positions(self, code):
        self.tree = self.parser.parse(bytes(code, "utf8"))
        root_node = self.tree.root_node
        self.identifier_positions = {}
        self.find_identifiers(root_node, code)
        return self.identifier_positions

    def get_ge_l(self, code):
        G, source = self.get_ast_nx(code, self.parser)
        # print(source)
        T = nx.dfs_tree(G, 0)


        # Copy node attributes from G to T
        for node in T.nodes():
            T.nodes[node]['label'] = G.nodes[node]['label']
            T.nodes[node]['features'] = G.nodes[node]['features']

        nodes = T.nodes()
        leaves = [x for x in T.nodes() if T.out_degree(x) ==
                        0 and T.in_degree(x) == 1]
        tokens_dict = {}
        tokens_type_dict = {}
        for leaf in leaves[:]:
            feature = G.nodes[leaf]['features']
            if feature.type == 'comment':
                leaves.remove(leaf)
                T.remove_node(leaf)
            else:
                start = feature.start_point
                end = feature.end_point
                token = self.index_to_code_token(start, end, source)
                # print('leaf: ', leaf, 'start: ', start,
                #     ', end: ', end, ', token: ', token)
                tokens_dict[leaf] = token
                tokens_type_dict[leaf] = feature.type
        assert len(leaves) == len(tokens_dict)
        sast = self.get_sast(T, leaves, tokens_dict, tokens_type_dict)

        # # Save AST graph to PNG
        # A = to_agraph(sast)
        # # print(G.nodes[0])

        # for node in A.iternodes():
        #     node_id = int(node.get_name())
        #     if sast.degree[node_id] == 1:  # If it's a leaf node
        #         node.attr['label'] = sast.nodes[node_id]['label']
        #     else:  # If it's not a leaf node
        #         node.attr['label'] = sast.nodes[node_id]['features'].type
        # A.layout('dot')
        # A.draw('SAST.png')
        # print(self.get_token_distance(leaves,sast))
        ge,l = self.calculate_global_efficiency(sast, leaves)

        return ge,l

    def process_examples(self):
        # exemplar_saving_dir
        if not os.path.exists(self.args.exemplar_saving_dir):
            os.makedirs(self.args.exemplar_saving_dir)
        exemplar_cache_pkl = os.path.join(
                        self.args.exemplar_saving_dir, 'exemplar_training.pkl')
        if os.path.exists(exemplar_cache_pkl):
            with open(exemplar_cache_pkl,'rb') as f:
                top_examples_heap = pickle.load(f)
            logger.info("Load cache exemplar from %s", exemplar_cache_pkl)
        else:
            # Use a min heap to keep the top examples
            top_examples_heap = []
            counter=count()
            if self.args.task == 'clone':
                self.examples = self.examples[:9010]
            for example in tqdm(self.examples, total=len(self.examples), desc="Getting top {} exemplas".format(str(self.args.exemplar_sample_num))):
                try:
                    if self.args.task=='generate':
                        ge,l= self.get_ge_l(example.target)
                    else:
                        if self.args.sub_task == 'php':
                            example.source = '<?php '+example.source
                        ge,l= self.get_ge_l(example.source)
                        if self.args.sub_task == 'php':
                            example.source = example.source[len('<?php '):]
                except RecursionError:
                    print(counter,'example too long too treat in ast')
                    continue
                if self.args.exemplar_strategy=='l':
                    value=l
                else:
                    value=ge
                # print(example.source,value)

                item=(value,next(counter),example)
                #for that when value ==last value, heapq will compare next item in tuple, but example can not be compared, so add count

                # If we have not yet found examplar_sample_num examples, we add the new one
                if len(top_examples_heap) < self.args.exemplar_sample_num:
                    heapq.heappush(top_examples_heap, item)
                    # min_value = min(min_value, value)
                else:
                    # If the new example's value is larger than the smallest value in the heap, we replace it
                    heapq.heappushpop(top_examples_heap, item)
                    # min_value, _ = top_examples_heap[0]
            with open(exemplar_cache_pkl,'wb') as f:
                pickle.dump(top_examples_heap,f)
                logger.info("Save cache exemplar to %s", exemplar_cache_pkl)
        # # Get the top examples from the heap, in descending order of value
        if self.args.task=='generate':
            top_examples = [{example.target:value} for value, _,example in sorted(top_examples_heap, reverse=True)]
        else:
            top_examples = [{example.source:value} for value, _,example in sorted(top_examples_heap, reverse=True)]
        print("Top 5 exemplars:",top_examples[:5])
        return [example for value, _,example in top_examples_heap]
    
    def get_ge_l_multiprocessing(self, example):
        if self.args.task=='generate':
            ge,l= self.get_ge_l(example.target)
        else:
            if self.args.sub_task == 'php':
                example.source = '<?php '+example.source
            ge,l= self.get_ge_l(example.source)
            if self.args.sub_task == 'php':
                example.source = example.source[len('<?php '):]
        return ge,l
    def process_examples_old(self):# tree_sitter_parser can not be multiprocessing
        # exemplar_saving_dir
        exemplar_cache_pkl = os.path.join(
                        self.args.exemplar_saving_dir, 'exemplar_training.pkl')
        if os.path.exists(exemplar_cache_pkl):
            with open(exemplar_cache_pkl,'rb') as f:
                top_examples = pickle.load(f)
            logger.info("Load cache exemplar from %s", exemplar_cache_pkl)
        else:
            with multiprocessing.Pool(self.args.cpu_count) as pool:
                results = []
                counter=count()
                for example, value_ in tqdm(zip(self.examples, pool.imap(self.get_ge_l_multiprocessing, self.examples)), total=len(self.examples), desc="Getting top {} exemplas".format(str(self.args.exemplar_sample_num))):
                    ge=value_[0]
                    l=value_[1]
                    if self.args.exemplar_strategy=='l':
                        value=l
                    else:
                        value=ge
                    item=(value,next(counter),example)
                    results.append(item)
                
                # 在主进程中使用 heapq.nlargest 获取最大的exemplar_sample_num个值
                top_examples = heapq.nlargest(self.args.exemplar_sample_num, results)
                with open(exemplar_cache_pkl,'wb') as f:
                    pickle.dump(top_examples,f)
                logger.info("Save cache exemplar to %s", exemplar_cache_pkl)
        # # Get the top examples from the heap, in descending order of value
        if self.args.task=='generate':
            top_examples_ordered = [{example.target:value} for value, _,example in sorted(top_examples, reverse=True)]
        else:
            top_examples_ordered = [{example.source:value} for value, _,example in sorted(top_examples, reverse=True)]
        print("Top 5 exemplars:",top_examples_ordered[:5])

        return [example for value, _,example in top_examples]
