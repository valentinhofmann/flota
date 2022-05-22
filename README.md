# FLOTA 

This repository contains the code and data for the ACL paper [An Embarrassingly Simple Method to Mitigate Undesirable 
Properties of Pretrained Language Model Tokenizers](https://aclanthology.org/2022.acl-short.43.pdf). 
The paper introduces FLOTA (Few Longest Token Approximation), a simple yet effective method
to improve the tokenization of pretrained language models.

# Dependencies

The code requires `Python>=3.8`, `numpy>=1.18`, `pandas>=1.1`, `torch>=1.2`, and `transformers>=4.12`.

# Data

The ArXiv challenge sets can be found in `data`.

# Usage

To replicate the main experiment, run the script `run_main.sh` in `src`.
To replicate the experiment on the impact of _k_, run the script `run_k.sh` in `src`.
To replicate the experiment with noisy input, run the script `run_noise.sh` in `src`.

The scripts expect the dataset files in `data`.

# Citation

If you use the code or data in this repository, please cite the following paper:

```
@inproceedings{hofmann2022flota,
    title = {An Embarrassingly Simple Method to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers},
    author = {Hofmann, Valentin and Sch{\"u}tze, Hinrich and Pierrehumbert, Janet},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
    year = {2022}
}
```