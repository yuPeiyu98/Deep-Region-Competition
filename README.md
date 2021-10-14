# Unsupervised Foreground Extraction via Deep Region Competition
<img src="teaser.png" alt="teaser" width="100%" />

[[Paper](TBD)] [[Code](https://github.com/yuPeiyu98/DRC)]

The official code repository for NeurIPS 2021 paper "Unsupervised Foreground Extraction via Deep Region Competition".

## Installation

The implementation depends on the following commonly used packages, all of which can be installed via conda.

| Package      | Version                          |
| ------------ | -------------------------------- |
| PyTorch      | ≥ 1.8.1                          |
| numpy        | *not specified* (we used 1.20.0) |

## Datasets and Pretrained Models

Datasets and pretrained models are available at: TBD

## Training

```bash
# Train a foreground extractor with specified checkpoint folder
python main.py --checkpoints <TO_BE_SPECIFIED>
```

You may specify the value of arguments. Please find the available arguments in the `config.yml.example` file in `drc_workspace` folder. 

## Testing

```bash
# Evaluate the extractor
python test.py --checkpoints <TO_BE_SPECIFIED>
```

## Citation

```
@inproceedings{yu2021unsupervised,
  author = {Yu, Peiyu and Xie, Sirui and Ma, Xiaojian and Zhu, Yixin and Wu, Ying Nian and Zhu, Song-Chun},
  title = {Unsupervised Foreground Extraction via Deep Region Competition},
  booktitle = {Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2021}
}
```
