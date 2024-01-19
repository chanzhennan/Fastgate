import os
import argparse

for i in range(32):
    os.system('python3 cublas-gemv.py --M %d --K %d --N %d' % (32 * (i + 1), 4096, 4096))
