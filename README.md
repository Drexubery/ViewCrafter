## ___***ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis***___
<div align="center">

 <a href='https://arxiv.org/abs/2310.12190'><img src='https://img.shields.io/badge/arXiv-2310.12190-b31b1b.svg'></a> &nbsp;
 <a href='https://doubiiu.github.io/projects/DynamiCrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/papers/2310.12190'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a> &nbsp;

_**[Wangbo Yu*](), [Jinbo Xing*](https://menghanxia.github.io), [Li Yuan*](), [Wenbo Hu&dagger;](https://wbhu.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Zhipeng Huang](), <br> [Xiangjun Gao](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en/), [Tien-Tsin Wong](https://www.cse.cuhk.edu.hk/~ttwong/myself.html), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate), [Yonghong Tian&dagger;]()**_
<br><br>

</div>

## 🔆 Introduction

ViewCrafter can generate high-fidelity novel views from <strong>a single or sparse reference image</strong>, while also supporting highly precise pose control. Below shows an example:


### Zero-shot novel view synthesis (single view)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Reference image</td>
        <td>Camera trajecotry</td>
        <td>Generated novel view video</td>
    </tr>

   <tr>
  <td>
    <img src=assets/train.png width="250">
  </td>
  <td>
    <img src=assets/ctrain.gif width="150">
  </td>
  <td>
    <img src=assets/train.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/wst.png width="250">
  </td>
  <td>
    <img src=assets/cwst.gif width="150">
  </td>
  <td>
    <img src=assets/wst.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/flower.png width="250">
  </td>
  <td>
    <img src=assets/cflower.gif width="150">
  </td>
  <td>
    <img src=assets/flower.gif width="250">
  </td>
  </tr>
</table>

### Zero-shot novel view synthesis (2 views)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Reference image 1</td>
        <td>Reference image 2</td>
        <td>Generated novel view video</td>
    </tr>

   <tr>
  <td>
    <img src=assets/car2_1.png width="250">
  </td>
  <td>
    <img src=assets/car2_2.png width="250">
  </td>
  <td>
    <img src=assets/car2.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/barn_1.png width="250">
  </td>
  <td>
    <img src=assets/barn_2.png width="250">
  </td>
  <td>
    <img src=assets/barn.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/house_1.png width="250">
  </td>
  <td>
    <img src=assets/house_2.png width="250">
  </td>
  <td>
    <img src=assets/house.gif width="250">
  </td>
  </tr>
</table>

## 📝 Changelog
- __[2024.08.28]__: Release pretrained models and code for single-view novel view synthesis.
- __[2024.08.28]__: Launch the project page and update the arXiv preprint.
<br>

## 🧰 Models

|Model|Resolution|Frames|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|:--------|
|ViewCrafter_25|576x1024|25| TBD (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt)|
|ViewCrafter_16|576x1024|16| TBD (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt)|


Currently, we provide two versions of the model: a base model that generates 16 frames at a time and an enhanced model that generates 25 frames at a time. The inference time can be reduced by using fewer DDIM steps.

## ⚙️ Setup

### 1. Clone ViewCrafter
```bash
git clone https://github.com/Drexubery/ViewCrafter.git
cd ViewCrafter
```
### 2. Installation

```bash
# Create conda environment
conda create -n viewcrafter python=3.9.16
conda activate viewcrafter
pip install -r requirements.txt

# Install PyTorch3D
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2
conda install pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2
rm -r pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

# Download DUSt3R
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

```

## 💫 Inference
### 1. Command line

Download pretrained model (ViewCrafter_25 for example) and put the `model.ckpt` in `checkpoints/model.ckpt`. \
Please refer to the [configuration document](docs/config_help.md) to set up camera trajectory and inference parameters.
```bash
  sh run.sh
```


### 2. Local Gradio demo

Download the pretrained model and put it in the corresponding directory according to the previous guidelines.
```bash
  python gradio_app.py 
```

