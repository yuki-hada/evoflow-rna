<!-- <div align="center"> -->

# EvoFlow-RNA 🧬 | A non-coding RNA masked diffusion model

[[Paper]](https://www.biorxiv.org/content/10.1101/2025.02.25.639942v2) [[Blog]](https://atombioworks.com/news/evoflow-rna-masked-discrete-diffusion/)

![til](./media/Fig1-overview.png)

The inference suite for`EvoFlow-RNA`, a high-throughput model capable of generating novel, diverse and naturalistic non-coding RNA sequences. In this version, we only release the mini-sized model (33m). A future version will include additional model sizes and further functionality.

## Installation

### CUDA 12.5 is required
We use `CUDA 12.5` and `Python 3.11` for this project. An additional key dependency is Flash-Attention v2. You can install our environment via `conda`.

```bash
# clone project
git clone https://url/to/this/repo/evoflow-rna.git
cd evoflow-rna

# create conda environment
conda env create -f env.yml

# activate environment
conda activate evoflow-rna

```

If any issues pertaining to `hydra` arise after setting up the environment, just uninstall `hydra-core` and re-install (including dependencies such as `hydra-colorlog`).

NOTE: We will be adding a docker image soon to this repository for streamlined setup!

## Example

See the `example.ipynb` notebook for usage instructions, such as how to get embeddings and generate non-coding RNAs, both conditionally and unconditionally!

## Weights

First download EvoFlow-RNA mini weights (about 0.4GB) and store them in the `~/weights` path for example notebook compatability.
```bash
cd weights
wget https://zenodo.org/records/15009560/files/mini-v1.ckpt
```
## License

This repository is kept under a `CC-BY-NC 4.0 International license`. All non-commercial projects are permitted to freely use, adopt, or build upon this work. See LICENSE.md for additional information.

## Acknowledgements

We modeled our repository off of [bytedance/dplm](https://github.com/bytedance/dplm) and their original work in protein sequence design with masked denoising discrete diffusion. We also appreciate the [RiNALMo](https://github.com/lbcb-sci/RiNALMo) team for their well-maintained project which we use as a static dependency. Please check them out!

## Citations

Please cite our code if used in your work!

`
@article{patel2025evoflow,
  title={EvoFlow-RNA: Generating and Representing non-coding RNA with a Language Model},
  author={Patel, Sawan and Peng, Fred Zhangzhi and Fraser, Keith and Friedman, Adam D and Chatterjee, Pranam and Yao, Sherwood},
  journal={bioRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
`
