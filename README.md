# SURGEON: Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity

**[CVPR 2025 Highlight]**

[📄 Paper](https://arxiv.org/abs/2503.20354) | [⚙️ Installation](#-installation) | [🚀 Usage](#-usage) | [📚 Citation](#-citation)

---

## 🔍 Overview

**SURGEON** is a fully test-time adaptation (FTTA) method designed to be lightweight and highly adaptable. It reduces memory cost during deployment on resource-constrained devices—without compromising accuracy or requiring any architectural changes or retraining.

We introduce a novel **Dynamic Activation Sparsity** mechanism that prunes activations on a per-layer basis using two adaptive metrics:

- **Gradient Importance (GI)**: Measures each layer’s contribution to accuracy.
- **Layer Activation Memory (LAM)**: Measures per-layer memory usage.

This allows SURGEON to **dynamically balance memory and adaptation performance** during test-time.

> 🔥 SURGEON achieves state-of-the-art performance on various datasets and architectures, while requiring significantly less memory than existing methods.

---

## 📦 Installation

We follow the environment setup from [RobustBench](https://github.com/RobustBench/robustbench) and [TTA baselines](https://github.com/mariodoebler/test-time-adaptation).

```bash
conda update conda
conda env create -f environment.yml
conda activate surgeon
```

---

## 📂 Datasets

We evaluate SURGEON on the following commonly-used corrupted datasets:

- [ImageNet-C](https://zenodo.org/records/2235448)
- [CIFAR-10C](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1)
- [CIFAR-100C](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1)

After downloading, update the dataset path in `./conf.py`:

```python
_C.DATA_DIR = "/your/data/path"
```

Recommended directory structure:

```
/your/data/path/
├── imagenet-c/
├── CIFAR-10-C/
└── CIFAR-100-C/
```

---

## 💾 Pretrained Models

Pretrained weights will be released soon. Stay tuned!

---

## 🚀 Usage

We provide config files for all baseline and SURGEON experiments in `./cfgs`.

To run an experiment:

```bash
python main.py --cfg ./cfgs/surgeon_cifar10.yaml
```

You can switch to other configs for different datasets or baselines.

---

## 📚 Citation

If you find our work useful, please cite us:

```latex
@inproceedings{ma2025surgeon,
  title={SURGEON: Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity},
  author={Ke Ma and Jiaqi Tang and Bin Guo and Fan Dang and Sicong Liu and Zhui Zhu and Lei Wu and Cheng Fang and Ying-Cong Chen and Zhiwen Yu and Yunhao Liu},
  booktitle={CVPR},
  year={2025}
}
```

---

## 🙌 Acknowledgements

This repo is built upon:

- [Test-Time Adaptation](https://github.com/mariodoebler/test-time-adaptation)  
- [RobustBench](https://github.com/RobustBench/robustbench)

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📬 Contact

If you encounter any problems or have questions, feel free to reach out:

📧 2544552413@mail.nwpu.edu.cn
