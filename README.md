# GSFTC: Graph-Spectral Filtering-enhanced Tensor Completion

This repository hosts the official implementation of **GSFTC**, a novel framework for high-dimensional tensor recovery.  
The code serves as the official implementation of the algorithm described in the paper published in *Advanced Engineering Informatics*.

## Contents

- `GSFTC.m` — main algorithm
- `helpers/` — tensor utilities
- `synthetic_demo.m` — quick demo script
- `dataset/`— datasets

## Usage
Run in MATLAB:
```matlab
synthetic_demo
```

## GNN model

The GNN models are trained and tested in accordance with the implementation in https://github.com/HazeDT/PHMGNNBenchmark.

## Citation
```latex
@article{ZHONG2026104013,
title = {Graph spectral filtering-enhanced tensor completion based on Schatten-p norm for missing measurements recovery with noise},
journal = {Advanced Engineering Informatics},
volume = {69},
pages = {104013},
year = {2026},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2025.104013},
url = {https://www.sciencedirect.com/science/article/pii/S1474034625009061},
author = {Zhirong Zhong and Xiaoguang Zhang and Xuanhao Hua and Zhi Zhai and Meng Ma and Jinxin Liu},
}
```
## License

Released under the MIT License. See `LICENSE` for details.
