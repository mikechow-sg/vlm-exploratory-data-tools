# Illusions of Generalizability, Realities of Volatility: A Case for Generative Vision Language Models as Exploratory Data Tools in Visual Content Analysis

## Abstract

> Automated visual content analysis has typically been dominated by task-specific deep-learning computer vision annotators. Generative vision language models (VLMs) such as Qwen2.5-VL have recently emerged as the latest technological development that could potentially transform computational visual analysis once more. Drawing from the growing use of large language models (LLMs) for text annotation, VLMs’ multimodality and promptability similarly offer a promise that any computational visual analysis task can be performed simply with a text prompt, thereby making the method more efficient, accessible, and versatile.
> 
> However, it is still unclear how generative VLMs perform in practice. This thesis seeks to explore the role of generative VLMs in visual content analysis. Through comparative applied experiments, this thesis disputes the convenient illusion that VLMs are generalizable and time-efficient. VLMs not only fail to achieve human-like understanding of visual semiotics, but their performance is highly erratic and sensitive to minor changes in prompts. However, their volatility conversely proved to be an asset in exploring datasets and exposing loopholes in assumptions. Consequently, I call for a reframing of VLMs — away from being annotators expected to be accurate and stable, but towards being exploratory data tools that augment human coders in the research design process.

---

## Contents

- `Notebook_1_shot_scale_pose_estimation.ipynb`: Notebook for shot scale classification using pose estimation by [Cao et al. (2017)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
- `Notebook_2_main_data_analysis.ipynb`: Notebook for the bulk of data analysis of annotations
- `Notebook_3_Experiment_2_VLM_setting_text_analysis.ipynb`: Notebook for text analysis of generative annotations
- `Notebook_4_Experiment_2_Test_Coding_1.ipynb`: Notebook for analysis after the first round of test coding
- `Notebook_5_Experiment_2_VLM_test_coding_2.ipynb`: Notebook for analysis after the second round of test coding
- `vlm_annotate.py`: Code used to run VLM annotation over a folder of shot thumbnails
- `input_data/`: Annotation data before analysis
- `output/`: Any output or intermediate data

## Installation

Clone this repository:

```bash
git clone https://github.com/YOUR-Username/YOUR-Repo-Name.git
cd vlm-exploratory-data-tools

---
```
This repository is in support of an MA Thesis project at the University of Amsterdam under the Media Studies Department. It was supervised by Dr. Bernhard Rieder. Please read the main paper for more details!
