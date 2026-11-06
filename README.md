# Learning without Forgetting for Vision-Language Models



<div align="center">
    <div>
        <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a><sup>1</sup>&emsp;
        <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang</a><sup>2</sup>&emsp;
        Yan Wang<sup>1</sup>&emsp;
        <a href='https://jingyinju.github.io/' target='_blank'>Jingyi Ning</a><sup>1</sup>&emsp;
        <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a><sup>1</sup>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhandc' target='_blank'>De-Chuan Zhan</a><sup>1</sup>&emsp;
        <a href='http://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>2</sup>
    </div>
    <div>
    <sup>1</sup>School of Artificial Intelligence, State Key Laboratory for Novel Software Technology, Nanjing University<br>
    <sup>2</sup>S-Lab, Nanyang Technological University
    </div>
</div>




<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=zhoudw-zdw.Proof&left_color=yellow&right_color=purple" alt="visitors">
  <a href="https://arxiv.org/abs/2305.19270">
    <img src="https://img.shields.io/badge/TPAMI2025-red" alt="arXiv">
  </a>

  
</div>



The code repository for "[Learning without Forgetting for Vision-Language Models](https://arxiv.org/abs/2305.19270) (**TPAMI 2025**)"  in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

```bibtex
@article{zhou2025learning,
  title={Learning without Forgetting for Vision-Language Models},
  author={Da-Wei Zhou and Yuanhan Zhang and Yan Wang and Jingyi Ning and Han-Jia Ye and De-Chuan Zhan and Ziwei Liu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  volume={47},
  number={6},
  pages={4489-4504},
}
```



# 📢 **Updates**

[07/2025] Check out our [latest work](https://arxiv.org/abs/2503.08510) on class-incremental learning with CLIP (**ICCV 2025**)! 

[02/2025] Accepted to TPAMI.

[10/2024] Code has been released.

[05/2023] [arXiv](https://arxiv.org/abs/2305.19270v1) paper has been released.


# 📝 Introduction

Class-incremental learning (CIL) aims to adapt to emerging new classes without forgetting old ones. Traditional CIL models are trained from scratch to continually acquire knowledge as data evolves.While traditional CIL methods focus on visual information to grasp core features, recent advances in Vision-Language Models (VLM) have shown promising capabilities in learning generalizable representations with the aid of textual information. However, when continually trained with new classes, VLMs often suffer from catastrophic forgetting of former knowledge. Applying VLMs to CIL poses two major challenges: 1) how to adapt the model without forgetting; and 2) how to make full use of the multi-modal information. To this end, we propose PROjectiOn Fusion (PROOF) that enables VLMs to learn without forgetting. To handle the first challenge, we propose training task-specific projections based on the frozen image/text encoders. When facing new tasks, new projections are expanded, and former projections are fixed, alleviating the forgetting of old concepts. For the second challenge,  we propose the fusion module to better utilize the cross-modality information. By jointly adjusting visual and textual features, the model can capture better task-specific semantic information that facilitates recognition. Extensive experiments on nine benchmark datasets with various continual learning scenarios and various VLMs validate that PROOF achieves state-of-the-art performance.

<div align="center">
<img src="resources/img.png" width="95%">
</div>

## 🔧 Requirements

**Environment**

1 [torch 1.11.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.12.0](https://github.com/pytorch/vision)

3 [open-clip 2.17.1](https://github.com/mlfoundations/open_clip/releases/tag/v2.17.1)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download.
- **Cars**: Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM)
- **UCF**: Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr)
- **Aircraft**: Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP)
- **Food**: Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia)
- **SUN**: OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV)
- **TV100**: [link](https://tv-100.github.io/)

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./utils/data.py` according to your own path. 

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```

## 🎈 Acknowledgement

This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [PyCIL](https://github.com/G-U-N/PyCIL). 

## 💭 Correspondence

If you have any questions, please  contact me via [email](mailto:zhoudw@lamda.nju.edu.cn) or open an [issue](https://github.com/zhoudw-zdw/Proof/issues/new).


## SECA class orders and forgetting

Install `PyYAML` in the existing PROOF environment (`pip install PyYAML`). Run
from the repository root:

```bash
python main.py --config exps/cifar_seca.json
```

This config differs from `exps/cifar.json` only by these two fields:

```json
"class_order_file": "class_orders/cifar100_order.yaml",
"class_order_idx": [0, 1, 2]
```

Each order runs with a fresh model, using the original seed list and all original
PROOF training settings. An explicit order overrides `shuffle`; omitting
`class_order_file` retains the original random-order behavior. A single integer
or a subset such as `[1, 2]` can select individual runs. Do not substitute three
random seeds for the three supplied orders. PROOF's existing `_set_random()`
behavior is retained.

The three YAML files are copied byte-for-byte from
[SECA_AAAI2026](https://github.com/HHHLF/SECA_AAAI2026/tree/84e6c14702bac6b311b5dc61f2b7a839ecb2acb5/class_orders)
(commit `84e6c14702bac6b311b5dc61f2b7a839ecb2acb5`). For ImageNet-R use
`class_orders/imagenet_R_order.yaml` with PROOF's `imagenetr`; for ImageNet-A use
`class_orders/imagenet_A_order.yaml` with `imageneta`. Add the same two fields to
your existing experiment JSON without changing its other settings. CIFAR100 is
called `cifar224` in PROOF. CIFAR IDs follow the standard CIFAR100 labels;
ImageNet IDs require identical ImageFolder class-to-index mappings and identical
train/test splits across the projects. Numeric permutations alone cannot verify
that separately prepared ImageNet datasets match.

Each run writes `..._order0.log` / `..._order0.metrics.jsonl` (and order1/order2)
under the original `logs/<model>/<dataset>/<init>/<increment>/` directory. The
JSONL header records the seed and full applied order. Subsequent records contain
`task`, `acc`, `acc_per_task`, `fg`, and `seca_fg`; the last task's values are the
final forgetting results. Rerunning a run replaces its JSONL file.

`acc_per_task` uses unrounded top-1 percentages from the existing class-incremental
predictions over **all seen classes**, partitioned by actual task boundaries
(including a larger initial task or shorter final task). No extra inference,
task-label masking, loss, training, rehearsal, transform, optimizer, or task-size
change is introduced.

For T completed tasks, standard `fg` is
`mean_j(max_{i=j,...,T-2} A[i,j] - A[T-1,j])`, for `j=0,...,T-2`,
in percentage points; it is zero for the first task. Negative values are retained.
`seca_fg` instead exactly reproduces SECA's pinned `continuum==1.2.7` implementation:
it sums only `j=0,...,T-3`, takes maxima over `i=j,...,T-3`, and divides by `T-1`.
That implementation omits the most recent old task and the most recent previous
evaluation; its second-task value is always zero. Use `seca_fg` against SECA's
existing `forgetting` logs, and use `fg` when reporting the standard definition.
Keep these two definitions separate.

CPU regression checks (no datasets or pretrained weights needed):

```bash
python -m unittest discover -s tests -v
```
