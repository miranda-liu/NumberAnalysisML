import gzip
import pandas as pd

with gzip.open('mnist.pkl.gz','rt') as f:
    for line in f:
        print('got line', line)
        f.readline()



