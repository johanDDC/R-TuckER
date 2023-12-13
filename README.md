## R-TuckER: Riemann optimization on manifolds of tensors for knowledge graph completion

This repository contains PyTorch implementation of R-TuckER model for knowledge graph link prediction task. 

### Summary

The proposed method is a modification of the approach described in paper [1]. It represents the knowledge graph as a tensor with a fixed SFT-rank and uses the Riemannian optimization approach for training. See details in [2].

### Repository structure

There are two types of models:

* `asymetric` -- model uses a regular Tucker decomposition for representing the knowledge graph, with distinct embeddings for the subjects and objects of each fact (fundamental records of the knowledge graph).
* `symmetric` -- model shares subjects and objects embeddings and utilizes SF-Tucker decomposition.

Note that unlike TuckER model, this approach doesn't employ such common DL techniques as Dropout or BatchNormalization.

### Link prediction results

Dataset | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---:
WN18RR | 0.477 | 0.544 | 0.492 | 0.446
FB15k-237 | 0.329 | 0.502 | 0.359 | 0.242

### Running a model

All main hyperparameters may be setuped in file `configs/base_config.py`. There are also som command line properties:

* `mode` eigher `symmetric` or `asymetric`
* `seed` random seed
* `nw` num_workers
* `device` either `cpu` or `cuda`
* `optim` either `rgd`, `rsgd` or `adam`
* `data` path to dataset

To reproduce the results from the paper, use the following combinations of hyperparameters with `batch_size=512`:

dataset   | rank (rel, ent) | lr    | lr_decay | reg_strategy | reg_init | reg_finish | reg_steps | momentum | label_smoothing | num_epochs
:---      | :---:     | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
WN18RR    | (10, 200) | 2000  | .9981 | "exp" | 1e-4  | 3e-9  | 350   | 0.8   | 0.1 | 1450
FB15k-237 | (200, 20) | 2000  | .9981 | "exp" | 1e-4  | 1e-10 | 100   | 0.8   | 0.1 | 1450

Also use the following command line options:

`python train.py --mode asymmetric --nw <NW> --seed 322 --data data/<dataset>/ --optim rsgd`

### Requirements

This implementation requires the following packages

>numpy>=1.23.1  
torch==1.13.1

Also custom library [`tucker_riemopt`](https://github.com/johanDDC/tucker_riemopt) is required.

### References

[TuckER: Tensor Factorization for Knowledge Graph Completion](https://arxiv.org/pdf/1901.09590.pdf)  
[TBA]

### License

MIT License
