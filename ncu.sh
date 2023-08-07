source /home/hongke21/nfs/miniconda3/bin/activate
conda activate torch116
/home/eva_share/opt/cuda-11.6/nsight-compute-2022.1.0/nv-nsight-cu-cli --set roofline --profile-from-start 0 -o gemv-0807-tensorcore-check --details-all python3 gemv-test.py