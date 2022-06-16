# BARC
[[Project Page](https://barc.is.tue.mpg.de/)] 


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
    * [Environment](#environment)
    * [Data Preparation](#data-preparation)
    * [Configurations](#configurations)
  * [Usage](#usage)
    * [Demo](#demo)
    * [Training](#training)
    * [Inference](#inference)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)



## Description

**B**reed **A**ugmented **R**egression using **C**lassification (BARC) is a method for dog pose and shape estimation.


## Installation

### Environment

The code has been tested with CUDA 10.1, CuDNN 7.5, Python 3.7 and PyTorch 1.6.0. 
```shell
    conda create -n "conda_barc" python=3.7.6 ipython
    conda activate conda_barc
    conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

To install the remaining dependencies run the following command:
```shell
    pip install -r requirements.txt
```


### Data Preparation

All necessary data be downloaded [here](https://owncloud.tuebingen.mpg.de/index.php/s/Pw2yoWnAmwcDb9S). A folder named 'checkpoint' contains pretrained models, copy it to the main folder of this project. A folder named 'stanext_related_data.zip' contains information related to the dataset. Please extract the zip file and copy it to data/stanext_related_data.

Download the Stanford Extra image dataset from https://github.com/benjiebob/StanfordExtra and store it in datasets/StanfordExtra_V12. 

Your folder structure should look as follows:
```bash
folder
├── data
│   └── breed_data
│   └── smal_data
│   └── statistics
│   └── stanext_related_data
├── datasets
│   └── test_image_crops
│   ├── StanfordExtra_V12
│   │   └── StanExtV12_Images
│   │   └── labels
├── scripts
│   └── ...
├── src
│   └── ...
```

### Configurations

All configuration files can be found in src/configs. You will need to adjust paths in barc_cfg_{train,test,visualization}.yaml and dataset_path_configs.py. If desired you can change the weights for different loss functions used at training time, see barc_loss_weights.json. I do not recommend changing zero-value weights to non-zero values, as most of the unused loss functions were removed.



## Usage

### Demo
```shell
    python scripts/visualize.py --workers 12  \
    --model-file-complete barc_complete/model_best.pth.tar \
    --config barc_cfg_test.yaml \
    --save-images True
```

### Training
```shell
    python scripts/train.py --workers 12 --checkpoint barc_new_v2 \
    train \
    --model-file-hg barc_hg_pret/checkpoint.pth.tar \
    --model-file-3d barc_normflow_pret/checkpoint.pth.tar
```

### Inference
```shell
    python scripts/test.py --workers 12  \
    --model-file-complete barc_complete/model_best.pth.tar \
    --config barc_cfg_visualization.yaml
```

## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```bibtex
@inproceedings{BARC:2022,
    title = {BARC: Learning to Regress 3D Dog Shape from Images by Exploiting Breed Information},
    author = {Rueegg, Nadine and Zuffi, Silvia and Schindler, Konrad and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```

## License

Software Copyright License for non-commercial scientific research purposes.
Please read carefully the following [terms and conditions](LICENSE) and any accompanying
documentation before you download and/or use BARC data, model and
software, (the "Data & Software"), including 3D meshes, images, videos,
textures, software, scripts, and animations. By downloading and/or using the
Data & Software (including downloading, cloning, installing, and any other use
of the corresponding github repository), you acknowledge that you have read
these [terms and conditions](LICENSE), understand them, and agree to be bound by them. If
you do not agree with these [terms and conditions](LICENSE), you must not download and/or
use the Data & Software. Any infringement of the terms of this agreement will
automatically terminate your rights under this [License](LICENSE).

## Acknowledgments


## Contact

The code of this repository was implemented by [Nadine Rüegg](mailto:nadine.rueegg@tuebingen.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).

