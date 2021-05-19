# User-Difference-Attention

Implementation of our paper "[UDA: A User-Difference Attention for Group Recommendation](https://www.sciencedirect.com/science/article/pii/S0020025521004254)". Our implementation is based on [Attentive-Group-Recommendation](https://github.com/LianHaiMiao/Attentive-Group-Recommendation).

**Please cite our paper if you use our codes. Thanks!**

BibTeX:

```
@article{ZAN2021401,
title = {UDA: A user-difference attention for group recommendation},
journal = {Information Sciences},
volume = {571},
pages = {401-417},
year = {2021},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.04.084},
author = {Shuxun Zan and Yujie Zhang and Xiangwu Meng and Pengtao Lv and Yulu Du},
}
```

## Getting Started

### 1. Environment Settings

- python 3.6
- basic python packages

```bash
pip install tqdm==4.59.0 scipy==1.5.4 numpy==1.16.4
```

- pytorch 1.6.0+cu101 (GPU version. Take `cu101` for example.)

```bash
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Examples to run the models

**(1) ML100K dataset**

uda:

```bash
python uda_main.py --dataset ml100k_1000_3 --embedding_size 128
```

agree:

```bash
python agree_main.py --dataset ml100k_1000_3 --embedding_size 128
```

**(2) Last.fm dataset**

uda:

```bash
python uda_main_lastfm.py --dataset lastfm_2 --embedding_size 128 --num_negatives 5  --lr "[1e-3,5e-4,1e-4]" 
```

agree:

```bash
python agree_main_lastfm.py --dataset lastfm_2 --embedding_size 128 --num_negatives 5  --lr "[1e-3,5e-4,1e-4]"
```

**(3) CAMRa2011 dataset** can be found in [Attentive-Group-Recommendation](https://github.com/LianHaiMiao/Attentive-Group-Recommendation).

## Related Projects

I also contribute to these open-source projects:

- [DeepCTR (Core Dev)](https://github.com/shenweichen/DeepCTR)  [![Downloads](https://pepy.tech/badge/deepctr)](https://pepy.tech/project/deepctr)
- [DeepCTR-Torch (Core Dev)](https://github.com/shenweichen/DeepCTR-Torch)  [![Downloads](https://pepy.tech/badge/deepctr-torch)](https://pepy.tech/project/deepctr-torch)
- [Attentive-Group-Recommendation](https://github.com/LianHaiMiao/Attentive-Group-Recommendation)
- [pytorch-doc-zh](https://github.com/apachecn/pytorch-doc-zh)

