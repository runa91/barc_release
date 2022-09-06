##  conda activate py397 ? barc ?
python3 scripts/train_hg.py \
    --arch=hg8 \
    --image-path=datasets/StanfordExtra_V12/StanExtV12_Images/ \
    --checkpoint=checkpoint/hg2 \
    --epochs=220 \
    --train-batch=6 \
    --test-batch=6 \
    --lr=5e-4 \
    --schedule 150 175 200