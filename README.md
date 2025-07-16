<div align="center">

![thinking-spongebob](figs/lava_fig2.png)

# 🎨 PrefPalette: Personalized Preference Modeling with Latent Attributes
  
[Shuyue Stella Li](https://stellalisy.com/), 
[Melanie Sclar](https://msclar.github.io/), 
[Hunter Lang](https://web.mit.edu/hjl/www/), 
[Ansong Ni](https://niansong1996.github.io/), 
[Jacqueline He](https://jacqueline-he.github.io/), 
[Puxin Xu](https://scholar.google.com/citations?user=VW1Bo1UAAAAJ&hl=en), 
[Andrew Cohen](https://scholar.google.com/citations?user=v1Frtb0AAAAJ&hl=en), 
[Chan Young Park](https://chan0park.github.io/), 
[Yulia Tsvetkov](https://homes.cs.washington.edu/~yuliats/), 
[Asli Celikyilmaz](http://asli.us/)
</div>

<div align="center">

[![Github](https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/stellalisy/LaVA)
[![Website](https://img.shields.io/badge/Site-000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://github.com/stellalisy/LaVA) 
[![Paper](https://img.shields.io/badge/Paper-000000.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://github.com/stellalisy/LaVA) 
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://github.com/stellalisy/LaVA)
</div>


## Setup

```sh
# Our codebase primarily uses Open-RLHF (https://github.com/OpenRLHF/OpenRLHF).
git clone git@github.com:stellalisy/PrefPalette
cd code

conda create -n lava python=3.10 
conda activate lava

pip install -r requirements.txt
pip install -e .
```

## Pipeline

### Counterfactual Generation
```sh
bash scripts/counterfactual_genetaion.sh
```

### Attribute Predictor Training
```sh
bash scripts/attribute_prediction.sh
```

### Latent-Attribute Preference Modeling
```sh
bash scripts/preferece_modeling.sh
```

## Ablations
<!-- ## Configurations

### Data
We include filtered and majority-labeled data in the paper. You may find a complete list in the `code/data` directory. For example, the ground truth data is termed `DeepScaleR`, and Llama 3.2 3B instruct labeled data, filtered to keep only the incorrect labels, is in the `DeepScaleR_mv_labeled_llama3.2_3b_instruct_incorrect` folder. You may change the data source by changing the variable `TASK` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

### Rewards
We include a list of rewards used in the paper below. Furthermore, note that for models without a chat template, be sure to add `_r1_only` as the suffix. You may change the reward function by changing the variable `REWARD` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

- `math`: Mathematical equivalence reward, which is the default
- `box_only_format`: Box-only formatting reward
- `contain_python_wo_backticks`: Mentioning of Python reward
- `random0.5`: Random reward with 50% returning 1


## Evaluations
To reproduce our evaluation results, use the following commands:

```sh
cd code

# For MATH-500 evaluation (requires NVIDIA A100 80GB PCIe for exact reproduction)
python scripts/eval_checkpoint.py --model_path Qwen/Qwen2.5-Math-7B --datasets MATH-500,AIME-2024,AIME-2025,AMC

# For MATH-500 evaluation matching our reported scores in wandb using checkpoints (requires NVIDIA H200 for exact reproduction)
python scripts/eval_checkpoint.py --model_path {} --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2
```

Note: To exactly reproduce `temperature = 0` results, both the GPU type and `--shards` parameter must match the original evaluation setup. This is because the batch size passed into VLLM can cause generation fluctuations. -->

## Paper

[TODO] Here's [the link](https://github.com/stellalisy/LaVA) to our paper.

## Citation

```bibtex
@misc{li2025lavapreferencemodeling,
      title={PrefPalette: Personalized Preference Modeling with Latent Attributes}, 
      author={Shuyue Stella Li and Melanie Sclar and Hunter Lang and Ansong Ni and Jacqueline He and Puxin Xu and Andrew Cohen and Chan Young Park and Yulia Tsvetkov and Asli Celikyilmaz},
      year={2025},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/XXXX.XXXXX}, 
}
```


## Acknowledgments
This repository is built based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). 
