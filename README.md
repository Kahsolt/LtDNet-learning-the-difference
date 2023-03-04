# LtDNet-learning-the-difference

    Learning the class-wise difference for image classification

----

Ablation on n_ref (total 10000 samples):

| n_ref | acc |
| :-: | :-: |
|  1 | 87.340% |
|  3 | 88.640% |
|  5 | 88.910% |
|  7 | 88.900% |
| 10 | 89.140% |
| 30 | 89.240% |

PGD attack (first 100 samples):

| n_ref | method | acc | asr |
| :-: | :-: | :-: | :-: |
| 1 | random | 52.941% | 47.059% |
| 3 | random | 33.333% | 66.667% |
| 5 | random | 27.451% | 72.549% |
| 7 | random | 23.529% | 76.471% |
| 1 | fixed  | 72.549% | 27.451% |
| 3 | fixed  | 56.863% | 43.137% |
| 5 | fixed  | 39.216% | 60.784% |
| 7 | fixed  | 35.294% | 64.706% |


### Quickstart

- `python preprocess.py`
- `python make_stats.py`
- `python train.py`
- `python test.py`
- `python attack.py`

----

by Armit
2023/02/22 
