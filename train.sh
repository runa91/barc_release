CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py --workers 12 --checkpoint model128 \
    --config barc_cfg_train.yaml \
    start \
    --model-file-hg barc_hg_pret/checkpoint.pth.tar \
    --model-file-3d barc_normflow_pret/checkpoint.pth.tar