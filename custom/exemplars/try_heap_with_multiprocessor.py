'''
# -*- coding: utf-8 -*-
Author: nchen
FilePath: /rethinking_code/custom/exemplars/try_heap_with_multiprocessor.py
'''
import heapq
import multiprocessing
from tqdm import tqdm
from itertools import count
def get_len(example):
    return example['0']


if __name__ == '__main__':
    examples= ['abcde','a','abc','ab','abcd']
    examples=[{'0':i} for i in examples]
    
    # top_heap=[]
    # for example in examples:
    #     print(example)
    #     if len(top_heap) < 2:
    #         # print(example,get_len(example))
    #         heapq.heappush(top_heap,(get_len(example),example))
    #     else:
            
    #         heapq.heappushpop(top_heap,(get_len(example),example))
    #         print(top_heap)
    with multiprocessing.Pool(4) as pool:
        results = []
        counter=count()
        for example, value in tqdm(zip(examples, pool.imap_unordered(get_len, examples)), total=len(examples)):
            item=(value,next(counter),example)
            #[('abcde', 0, {'0': 'abcde'}), ('abcd', 3, {'0': 'ab'})]
            #注意 这样加example很可能元素对不上！ 所以不能同时使用multiprocessor
            results.append(item)
        
        # 在主进程中使用 heapq.nlargest 获取最大的三个值
        top_two = heapq.nlargest(2, results)
        print(top_two)

