# DIMP

[Self-supervised Graph Neural Networks via **D**iverse and **I**nteractive **M**essage **P**assing.](http://yangliang.github.io/pdf/aaai22.pdf)


## Dependencies

- torch 1.7.1
- sklearn 0.21.3
- numpy 1.16.0
- scipy 1.5.4
- munkres 1.1.4

## Usage

Train and evaluate the node_classify model by executing
```
python -u execute.py --dataset cora --nb_epochs 2000 --patience 20 --numb_start 4 --numb_end 5 --chu_start 1 --chu_end 2 --chu_strip 0.5 --h1 -0.1 --h2 -0.1 --k_numb1 0 --k_numb2 1
```
The `--dataset` argument should be one of [ cora, citeseer, pubmed, amazon_electronics_computers, amazon_electronics_photo, ms_academic_cs, ms_academic_phy ].

Train and evaluate the graph_classify model by executing
```
python graph_classify/graph_classify.py
```
The `--dataset` argument should be one of [ MUTAG, PTC_MR, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K].
