'''
# -*- coding: utf-8 -*-
nchen
FilePath: /rethinking_code/custom/exemplars/try_multi_processor.py
'''
import multiprocessing
from tqdm import tqdm

def x2(a):
    return 2*a
if __name__ == '__main__':
    pool = multiprocessing.Pool(4)#args.cpu_count
    examples =[1,2,3,4,5]
    features = pool.map(x2, tqdm(
        examples, total=len(examples),desc="Convert examples to features"))
    print(features)