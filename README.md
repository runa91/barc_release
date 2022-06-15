# BARC
[[Project Page](https://barc.is.tue.mpg.de/)] 


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
    * [Dependencies](#dependencies)
    * [Preparing the data](#preparing-the-data)
  * [Usage](#usage)
    * [Demo](#demo)
    * [Training](#training)
    * [Inference](#inference)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)


## Usage

### Demo
```shell
    python scripts/visualize.py --workers 12  \
    --model-file-complete barc_new_v2/model_best.pth.tar \
    --config barc_cfg_test.yaml \
    --save-images True
```

### Training
```shell
    python scripts/train.py --workers 12 --checkpoint barc_new_v2 \
    train \
    --model-file-hg dogs_hg8_ksp_24_sev12_v3/model_best.pth.tar \
    --model-file-3d Normflow_CVPR_set8_v3k2_v1/checkpoint.pth.tar
```

### Inference
```shell
    python scripts/test.py --workers 12  \
    --model-file-complete barc_new_v2/model_best.pth.tar \
    -- --config barc_cfg_visualization.yaml
```

## Description

**B**reed **A**ugmented **R**egression using **C**lassification (BARC) is a method for dog pose and shape estimation.

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

