rm *.json
python scripts/test.py --workers 16  \
    --model-file-complete folorin/model_best.pth.tar \
    --config barc_cfg_test.yaml \
    --save-images True