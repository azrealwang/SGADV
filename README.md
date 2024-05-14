# Similarity-based Gray-box Adversarial Attack Against Deep Face Recognition [pdf](https://arxiv.org/pdf/2201.04011)

Hanrui Wang<sup>1</sup>, Shuo Wang<sup>2</sup>, Zhe Jin<sup>3</sup>, Yandan Wang<sup>4</sup>, Cunjian Chen<sup>5</sup>, Massimo Tistarelli<sup>6</sup>

The majority of adversarial attack techniques perform well against deep face recognition when the full knowledge of the system is revealed (white-box). However, such techniques act unsuccessfully in the gray-box setting where the face templates are unknown to the attackers. In this work, we propose a similarity-based gray-box adversarial attack (SGADV) technique.

<img src="figures/scenario.png" alt="scenario" style="width:400px;"/>

## Introduction

Run attack: SGADV.py

Objective function: foolbox/attacks/gradient_descent_base.py

New developed tools: foolbox/utils.py

## Extra tools 

Filter objects of CelebA: tools/fetch_celebAhq.py

Feature embeddings and save to .mat: tools/feature_embedding.py

## Citation
If using this project in your research, please cite our paper.
```
@inproceedings{wang2021similarity,
  title={Similarity-based Gray-box Adversarial Attack Against Deep Face Recognition},
  author={Wang, Hanrui and Wang, Shuo and Jin, Zhe and Wang, Yandan and Chen, Cunjian and Tistarelli, Massimo},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```
## Note
The code in the folder *foolbox* is derived from the project [foolbox](https://github.com/bethgelab/foolbox).

Images in the folder *data* are only examples from [LFW](http://vis-www.cs.umass.edu/lfw/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
