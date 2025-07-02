# Multi-Modal-Base

[![Demo](https://img.shields.io/badge/Web-Demo-blue?logo=google-chrome)](https://github.com/WangJingyao07/C3R.github.io)
![Static Badge](https://img.shields.io/badge/ICML25-yellow)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Stars](https://img.shields.io/github/stars/WangJingyao07/Multi-Modal-Base)

An open source multi-modal codebase

This repository provides Pytorch implementation of MML baselines and configurations for academic research, as well as several plug-and-play MML modules, including PyTorch code for [ICML2025] Towards the Causal Complete Cause of Multi-Modal Representation Learning.



### Quick Start

```bash
# create env
conda create -n mm_base python=3.9 -y
conda activate mm_base

# install deps
pip install -r requirements.txt          # we provide a template
# OR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm scikit-learn opencv-python librosa pandas tqdm einops sentencepiece wandb
...
````

> **Tip **The code uses single-card training by default; if multiple cards are required, directly `torchrun --nproc_per_node=N train/...py ...`.





### Directory Layout

```text
├── backbone/           # BERT / GPT / ViT / ResNet wrappers
├── data/               # PyTorch Dataset & Dataloader helpers
├── dataset/            # Place raw / pre-processed datasets here
├── models/             # Fusion baselines + MLU / MM-Pareto / OGM
├── prepross/           # Audio ?? Text ?? Image preprocessing scripts
├── train/              # All training entry points (per-task)
├── utils/              # logger, metrics, schedulers
└── saved/              # checkpoints & TensorBoard logs (auto-created)
```





### Dataset Preparation

The following table summarizes the default `--data_path` of the script, which can be modified to any absolute path as required.

*Please note that the files still retain the original server-side data paths (unchanged); you may modify them as needed.*

| Dataset              | Download                                                                                  | Temp                   | Pre-Process                             |
| ---------------- | ------------------------------------------------------------------------------------- | ---------------------- | --------------------------------- |
| **IEMOCAP**      | [USC Release](https://sail.usc.edu/iemocap/)                                          | `dataset/IEMOCAP/raw/` | `prepross/preprocess_iemo.py`     |
| **CREMA-D**      | [Zenodo](https://zenodo.org/record/1109496)                                           | `dataset/CREMAD/raw/`  | `prepross/gen_cre_txt.py`         |
| **MVSA-Single**  | [Official](https://mvsanet.github.io/)                                                | `dataset/MVSA_Single/` | *No additional scripts required*                          |
| **Food-101**     | [Kaggle](https://www.kaggle.com/datasets/kmader/food41)                               | `dataset/Food101/raw/` | `prepross/video_preprocessing.py` |
| **BRATS-2021**   | [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) | `dataset/BRATS/raw/`   | *Comes with NIfTI loader*                 |
| **NYU-Depth V2** | [Official](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)                  | `dataset/NYU/raw/`     | `prepross/gen_stat.py`            |

> Example script execution
>
> ```bash
> python prepross/preprocess_iemo.py \
>        --in_dir dataset/IEMOCAP/raw \
>        --out_dir dataset/IEMOCAP \
>        --split_ratio 0.8 0.1 0.1
> ```





### Training & Evaluation

We offer two ways to run our code:

**Way 1:** Train models in a parallel fashion using the carefully organized [`scripts`](https://github.com/WangJingyao07/Multi-Modal-Base/tree/main/run), which is as follows:

```bash
# Use bash interpreter directly

bash run/train_all_cluster_safe.bash
```

Or grant executable permissions first, then run as an executable file:

```bash
chmod +x run/train_all_cluster_safe.bash  
./run/train_all_cluster_safe.bash
```



**Way 2:** Directly write:

```bash
python ../train/train_food.py --batch_sz 16 --gradient_accumulation_steps 40 --savedir ./saved/food101 --name IB_VT02 --task food101  --task_type classification --model mml_vt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1 --noise 0.0
```



All scripts use `argparse`; run any of them with `-h` to see the full list of options. For example:

```bash
python train/train_food.py -h
```

Selected configurable parameters:

| Flag                            | Description                                                  | Default               |
| ------------------------------- | ------------------------------------------------------------ | --------------------- |
| `--model`                       | Choose *mml_vt / mml_avt / latefusion / mlu / mm_pareto* etc. | *mml_vt*              |
| `--bert_model`                  | Path to the language encoder or HuggingFace model name       | `./bert-base-uncased` |
| `--vit_model`                   | Vision backbone (e.g., `vit_base_patch16_224`)               | *resnet50*            |
| `--batch_size`                  | Training batch size                                          | *16*                  |
| `--max_epochs`                  | Number of training epochs                                    | *100*                 |
| `--gradient_accumulation_steps` | Gradient accumulation steps                                  | *24*                  |
| `--savedir`                     | Output directory for logs / weights                          | `./saved/`            |



### Citation

If you find our work and codes useful, please consider citing our paper and star our repository (🥰🎉Thanks!!!):

```bibtex
@misc{wang2025causal,
      title={Towards the Causal Complete Cause of Multi-Modal Representation Learning}, 
      author={Jingyao Wang and Siyu Zhao and Wenwen Qiang and Jiangmeng Li and Changwen Zheng and Fuchun Sun and Hui Xiong},
      year={2025},
      eprint={2407.14058},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14058}, 
}
```

