# GazeVLA: Learning Human Intention for Robotic Manipulation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.22615)
[![Project Page](https://img.shields.io/badge/Project%20Page-Website-yellow?style=flat&logo=googlechrome&logoColor=white)](https://gazevla.github.io/)
[![Video YouTube](https://img.shields.io/badge/Video-YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://youtu.be/a1-jS9uWwAI)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=flat&logo=github&logoColor=white)](https://github.com/lichy2004/GazeVLA)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

</details>

![](figures/dataset.jpg)
This is the human data processing script used in the GazeVLA paper. It supports hand motion alignment and 2D & 3D visualization for egocentric human datasets, and additionally integrates latest datasets including xperience-10m, VITRA, and EgoScale. All data has been standardized and unified into a consistent format, aiming to contribute to the research community.


## 📋 Table of Contents
- [🛠 Environment Setup](#-environment-setup)
- [💡 Example](#-example)
- [🙏 Acknowledgements](#-acknowledgements)
- [✍️ Citation](#️-citation)


## 🛠 Environment Setup

### Step 1: Clone the Repository
```bash
git clone git@github.com:lichy2004/GazeVLA-Data.git GazeVLA-Data
cd GazeVLA-Data
```

### Step 2: Set Up Python Environment
```bash
# Create a conda environment
conda create -n gazevla_data python=3.11 -y
conda activate gazevla_data

# Install requirements
pip install -r requirements.txt

# You can try installing PyTorch3D using the following method
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
```

### Step3: Download MANO model
Hand annotations in the MANO format can be downloaded after accepting their [license agreement](https://mano.is.tue.mpg.de/).

We requires the `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` files for loading and rendering of hand poses. These files can be obtained from the `mano_v1_2.zip` file located in the "Models & Code" section of the `MANO` website. After downloading, extract the zip file to your local disk, and the `*.pkl` files can be found at the following path: `assets/mano_v1_2/models`.

## 💡 Example
Each dataset must first be downloaded from the official website with the correct file structure. Refer to [DATA.md](./DATA.md) for instructions on downloading datasets and organizing the data. An example of data processing and visualization is shown below.

```bash
# processing
python src/dataset/EgoDex.py

# visualize
python examples/vis.py --file_path data/EgoDex/processed/00000000.hdf5
```



## TODO

The following features are planned for future implementation:

- [x] EgoDex
- [x] HOT3D
- [ ] HoloAssist
- [ ] OAKINK2
- [ ] TACO
- [ ] HOI4D
- [ ] H2O
- [ ] xperience-10m
- [ ] VITRA
- [ ] EgoScale



##  🙏 Acknowledgements

Our visualization code is built upon [Viser](https://viser.studio/main/) and our data processing scripts refer to the official code of each dataset.. These code serve as an essential foundation for our implementation, and we deeply appreciate the time, effort, and expertise they shared with the community.

## ✍️ Citation


If you find our work useful, please cite us:


```
@article{}
```

## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

<!-- *Chart updates automatically. Click to interact with the full timeline.* -->