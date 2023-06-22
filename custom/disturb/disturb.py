'''
# -*- coding: utf-8 -*-
nchen
FilePath: /rethinking_code/custom/disturb/disturb.py
'''

import random
from tqdm import tqdm
from tree_sitter import Language, Parser

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


class Disturb():
    def __init__(self, args,examples, disturb_percentage=100):
        self.args = args
        self.examples = examples
        self.lang =  get_lang_by_task(args.task, args.sub_task)
        LANGUAGE = Language('build/my-languages.so', self.lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        self.parser = parser
        self.disturb_percentage = disturb_percentage
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

    def get_disturbed_code(self, code):
        disturb_percentage=self.disturb_percentage
        identifier_positions = self.get_identifier_positions(code)
        identifier_positions_list = [(identifier, pos) for identifier, positions in identifier_positions.items() for pos in positions]
        identifier_positions_list.sort(key=lambda x: x[1])

        disturbed_code = code
        offset = 0
        total_identifiers = len(identifier_positions_list)
        disturb_count = int(total_identifiers * disturb_percentage / 100)
        disturb_indices = random.sample(range(total_identifiers), disturb_count)
        remaining_identifiers = [identifier_positions_list[i][0] for i in disturb_indices]

        for idx, (original_identifier, pos) in enumerate(identifier_positions_list):
            if idx in disturb_indices:
                new_identifier = random.choice(remaining_identifiers)
                remaining_identifiers.remove(new_identifier)
                disturbed_code = disturbed_code[:pos+offset] + new_identifier + disturbed_code[pos+offset+len(original_identifier):]
                offset += len(new_identifier) - len(original_identifier)

        return disturbed_code

    def process_examples(self):
        disturbed_examples= []
        for example in tqdm(self.examples, total=len(self.examples), desc="Disturbing Examples at {}%".format(self.disturb_percentage)):
            if self.args.task=='generate':
                example.target = self.get_disturbed_code(example.target)
            else:
                if self.args.sub_task == 'php':
                    example.source = '<?php '+example.source
                example.source = self.get_disturbed_code(example.source)
                if self.args.sub_task == 'php':
                    example.source = example.source[len('<?php '):]
            disturbed_examples.append(example)
        return disturbed_examples