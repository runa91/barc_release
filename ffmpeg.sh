## tar -xvf  out.tar -C .
# ffmpeg -y -threads 16 -i cat1_barc/comp_pred_%06d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error cat1_barc.mp4
# ffmpeg -y -threads 16 -i results/barc_complete/dog2w_mp4ImgCrops_vis_test_best_until_e199/keypoints_pred_%06d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error dog2w_keys.mp4
ffmpeg -y -threads 16 -r 10 -i /data/datasets/tmpcat1/Cat_black_move_sky6_Camera4_%06d_color.jpg -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error -r 10 catbms6.mp4
