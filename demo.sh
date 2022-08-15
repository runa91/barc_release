    # python scripts/visualize.py --workers 12  \
    # --model-file-complete barc_complete/model_best.pth.tar \
    # --config barc_cfg_visualization.yaml  \
    # --image_path 'datasets/ins/cat1_mp4/'

    # python scripts/visualize.py --workers 12  \
    # --model-file-complete barc_complete/model_best.pth.tar \
    # --config barc_cfg_visualization.yaml  \
    # --image_path '/data/datasets/MP4for_test/White_boxer_running_laps/'
    # ## --image_path '/data/851/tmp/dog3long_mp4/'
    
    ## test custom model. input video.save img and video in the same path.
CUDA_VISIBLE_DEVICES=3    python scripts/visualize.py --workers 24  \
    --model-file-complete barc_complete/model_best.pth.tar \
    --config barc_cfg_visualization.yaml  \
    --img_path '/data/851/barc_npet/datasets/barc_input/'
    # --img_path '/data/hxh/project/Coarse-to-fine-3D-Animal/data/StanfordExtra_v12/test_images/'