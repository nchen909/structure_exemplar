'''
# -*- coding: utf-8 -*-
Author: nchen
FilePath: /rethinking_code/custom/exemplars/try_heap.py
'''
import heapq
import multiprocessing

def get_len(example):
    return example['0']


if __name__ == '__main__':
    examples= ['abcde','a','abc','ab','abcd']
    examples=[{'0':i} for i in examples]
    
    top_heap=[]
    for example in examples:
        print(example)
        if len(top_heap) < 2:
            # print(example,get_len(example))
            heapq.heappush(top_heap,(get_len(example),example))
        else:
            
            heapq.heappushpop(top_heap,(get_len(example),example))
            print(top_heap)
    print(top_heap)
