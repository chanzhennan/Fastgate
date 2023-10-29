source ~/space/anaconda3/bin/activate
conda activate torch
/usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set full --profile-from-start 0 -o bs2-d5120-hd5120-1022 --details-all python3 profiling.py