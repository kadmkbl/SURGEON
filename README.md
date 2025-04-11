# [CVPR 2025 Highlight] SURGEON: Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity 
Despite the growing integration of deep models into mobile terminals, the accuracy of these models declines significantly due to various deployment interferences. Test-time adaptation (TTA) has emerged to improve the performance of deep models by adapting them to unlabeled target data online. Yet, the significant memory cost, particularly in resource-constrained terminals, impedes the effective deployment of most backward-propagation-based TTA methods. To tackle memory constraints, we introduce SURGEON, a method that substantially reduces memory cost while preserving comparable accuracy improvements during fully test-time adaptation (FTTA) without relying on specific network architectures or modifications to the original training procedure. Specifically, we propose a novel dynamic activation sparsity strategy that directly prunes activations at layer-specific dynamic ratios during adaptation, allowing for flexible control of learning ability and memory cost in a data-sensitive manner. Among this, two metrics, Gradient Importance and Layer Activation Memory, are considered to determine the layer-wise pruning ratios, reflecting accuracy contribution and memory efficiency, respectively. Experimentally, our method surpasses the baselines by not only reducing memory usage but also achieving superior accuracy, delivering SOTA performance across diverse datasets, architectures, and tasks.

Axriv: [https://arxiv.org/abs/2503.20354]

The complete code will be released soon.

---
## 🌐 **Citations**

**The following is a BibTeX reference:**

``` latex
@inproceedings{ma2025surgeon,
      title={SURGEON: Memory-Adaptive Fully Test-Time Adaptation via Dynamic Activation Sparsity}, 
      author={Ke Ma and Jiaqi Tang and Bin Guo and Fan Dang and Sicong Liu and Zhui Zhu and Lei Wu and Cheng Fang and Ying-Cong Chen and Zhiwen Yu and Yunhao Liu},
      year={2025},
      booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
