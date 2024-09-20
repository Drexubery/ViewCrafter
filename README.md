## ___***ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis***___
<div align="center">
<img src='assets/logo.png' style="height:100px"></img>

 <a href='https://arxiv.org/abs/2409.02048'><img src='https://img.shields.io/badge/arXiv-2409.02048-b31b1b.svg'></a> &nbsp;
 <a href='https://drexubery.github.io/ViewCrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://www.youtube.com/watch?v=WGIEmu9eXmU'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a>&nbsp;
 <a href='https://huggingface.co/spaces/Doubiiu/ViewCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

_**[Wangbo Yu*](https://scholar.google.com/citations?user=UOE8-qsAAAAJ&hl=zh-CN), [Jinbo Xing*](https://doubiiu.github.io/), [Li Yuan*](), [Wenbo Hu&dagger;](https://wbhu.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Zhipeng Huang](), <br> [Xiangjun Gao](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en/), [Tien-Tsin Wong](https://www.cse.cuhk.edu.hk/~ttwong/myself.html), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate), [Yonghong Tian&dagger;]()**_
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

### Zero-shot novel view synthesis (two views)
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

## 🗓️ TODO
- [x] [2024-09-01] Launch the project page and update the arXiv preprint.
- [x] [2024-09-01] Release pretrained models and the code for single-view novel view synthesis.
- [ ] Release the code for sparse-view novel view synthesis.
- [ ] Release the code for iterative novel view synthesis.
- [ ] Release the code for 3D-GS reconstruction.
<br>

## 🧰 Models

|Model|Resolution|Frames|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|:--------|
|ViewCrafter_25|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt)|
|ViewCrafter_16|576x1024|16| 18.3GB & 75s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_16/blob/main/model.ckpt)|
|ViewCrafter_25_512|320x512|25| 13.8GB & 50s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25_512/blob/main/model.ckpt)|


Currently, we provide three versions of the model: a base model that generates 16 frames at a time, an enhanced model that generates 25 frames at a time (used by default), and a low-resolution model that produces 25 frames of 320x512 video. The inference time can be reduced by using fewer DDIM steps.

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
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

# Download DUSt3R
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

```

## 💫 Inference
### 1. Command line

(1) Download pretrained model (ViewCrafter_25 for example) and put the `model.ckpt` in `checkpoints/model.ckpt`. \
(2) Run [inference.py](./inference.py) using the following script. Please refer to the [configuration document](docs/config_help.md) and [render document](docs/render_help.md) to set up inference parameters and camera trajectory. 
```bash
  sh run.sh
```

### 2. Local Gradio demo

Download the pretrained model and put it in the corresponding directory according to the previous guideline, then run:
```bash
  python gradio_app.py 
```

## 😉 Citation
Please consider citing our paper if our code is useful:
```bib
  @article{yu2024viewcrafter,
    title={ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis},
    author={Yu, Wangbo and Xing, Jinbo and Yuan, Li and Hu, Wenbo and Li, Xiaoyu and Huang, Zhipeng and Gao, Xiangjun and Wong, Tien-Tsin and Shan, Ying and Tian, Yonghong},
    journal={arXiv preprint arXiv:2409.02048},
    year={2024}
  }
```

<a name="disc"></a>
## 📢 Disclaimer
⚠️This is an open-source research exploration rather than a commercial product, so it may not meet all your expectations. Due to the variability of the video diffusion model, you may encounter failure cases. Try using different seeds and adjusting the render configs if the results are not desirable.
Users are free to create videos using this tool, but they must comply with local laws and use it responsibly. The developers do not assume any responsibility for potential misuse by users.
****

