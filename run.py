import os

for i in range(13):
    os.system('python3 edgemm-check.py --k 12288 --n %d' % (128 * (2 ** i)))