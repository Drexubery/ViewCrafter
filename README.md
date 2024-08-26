## ___***PVDiffusion: Point-conditioned Video Diffusion Models for High-fidelity Novel View Synthesis***___
<div align="center">

 <a href='https://arxiv.org/abs/2310.12190'><img src='https://img.shields.io/badge/arXiv-2310.12190-b31b1b.svg'></a> &nbsp;
 <a href='https://doubiiu.github.io/projects/DynamiCrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/papers/2310.12190'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a> &nbsp;

_**[Wangbo Yu](), [Jinbo Xing](https://menghanxia.github.io), [Li Yuan](), [Wenbo Hu](https://wbhu.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Zhipeng Huang](), <br> [Xiangjun Gao](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en/), [Tien-Tsin Wong](https://www.cse.cuhk.edu.hk/~ttwong/myself.html), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate), [Yonghong Tian]()**_
<br><br>

</div>

## 🔆 Introduction

PVDiffusion can generate high-fidelity novel views from <strong>a single or sparse reference image</strong>, while also supporting highly precise pose control. Below shows an example:


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
