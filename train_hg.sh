##  conda activate  barc 
CUDA_VISIBLE_DEVICES=1,2,3 \
python3 scripts/train_hg.py \
    --arch=hg2 \
    --image-path=datasets/StanfordExtra_V12/StanExtV12_Images/ \
    --checkpoint=checkpoint/hg2_seg \
    --epochs=220 \
    --train-batch=32 \
    --test-batch=32 \
    --lr=5e-4 
    # --schedule 150 175 200
    # --resume=checkpoint/barc_hg_pret/checkpoint.pth.tar \
