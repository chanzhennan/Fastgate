import os
import argparse

for i in range(1):
    i = 4
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 4096, 4096))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 4096, 11008))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 4096, 12288))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 4096, 13696))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 4096, 16384))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 11008, 4096))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 13696, 4096))
    os.system('python3 edgemm-check-tr.py --m %d --k %d --n %d' % (2 ** i, 16384, 4096))
