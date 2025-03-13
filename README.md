<h1>MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding</h1>

<div align="center">
    Zhoutong Ye, Mingze Sun, Huan-ang Gao, Chun Yu, Yuanchun Shi
</div>

<div align="center">
<a href="https://arxiv.org/abs/2503.09348" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-MOAT-red?logo=arxiv" height="20" />
</a>
<a href="https://cambrian-yzt.github.io/MOAT/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-MOAT-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/waltsun/MOAT" target="_blank">
    <img alt="HF Dataset: MOAT" src="https://img.shields.io/badge/%F0%9F%A4%97%20_HuggingFace-MOAT-yellow" height="20" />
</a>
<a href="https://github.com/Cambrian-yzt/MOAT" target="_blank">
    <img alt="GitHub: MOAT" src="https://img.shields.io/badge/GitHub-MOAT-yellow?logo=github" height="20" />
</a>
</div>

<img src=".\figures\fig1.png" alt="fig1" style="zoom:100%;" />

## Overview

**MOAT** (**M**ultimodal model **O**f **A**ll **T**rades) is a challenging benchmark for large multimodal models (LMMs). It consists of vision language (VL) tasks that require the LMM to integrate several VL capabilities and engage in human-like generalist visual problem solving. Moreover, many tasks in **MOAT** focus on LMMs' capability to ground complex text and visual instructions, which is crucial for the application of LMMs in-the-wild. Developing on the VL capability taxonomies proposed in previous benchmark papers, we define 10 fundamental VL capabilities in **MOAT**. 

<img src=".\figures\fig2.png" alt="fig2" style="zoom:100%;" />

Notably, we purposefully insulated **MOAT** from the influence of domain knowledge, text generation style, and other external factors by making the questions close-ended (i.e. have a single short answer) and solvable with the information and hints provided in the question itself. This allows **MOAT** to focus on fundamental generalist VL capabilities. We also did not include VL capabilities like *general object recognition* and *attribute recognition* in our taxonomy, since these are required by all **MOAT** tasks, and performance on these fronts can be reflected in the overall accuracy on **MOAT**.

## Benchmark Composition

**MOAT** tasks require LMMs to integrate up to 6 fundamental VL capabilities. We report the proportion of questions requiring each VL capability, the distribution of the number of VL capabilities required, and the 15 most common capability combinations required in **MOAT**.

<img src=".\figures\fig3.png" alt="fig3b" style="zoom:100%;" />



## Leaderboard

**ALL** existing LMMs, proprietary and open source, perform very poorly on **MOAT**, with the best performing model (OpenAI o1) achieving an accuracy (38.8%) less than half of that achieved by humans (82.7%). For individual VL capabilities, **CNT**, **RLA**, **3DTF** and **3DQNT** saw consistent poor performance by LMMs. In addition, **GNDT** and **GNDV** performance did not scale well with model size. Please refer to our paper for more detailed analysis of the results, as well as discussion on the implication of LMM architecture choices such as tiling and built-in CoT reasoning (or '*thinking*') capability.

<img src=".\figures\fig4.png" alt="fig4" style="zoom:100%;" />

## Usage

**Quickstart**

* Dependencies are listed in `./requirements.txt`. We used `Python 3.12.8` in our experiments. Run `pip install -r requirements.txt` to install all dependencies. 

* The dataset is available on [Hugging Face](https://huggingface.co/datasets/waltsun/MOAT). Our code will automatically download the dataset from Hugging Face.

* To evaluate an LMM, run `python main.py`, and the results will be logged under `./logs/`. Change the model name, API endpoint, and API key in `./configs/constants.py`.

**Run Your Own Evaluation**

You can access our dataset with the following code:

```python
from datasets import load_dataset
dataset = load_dataset("waltsun/MOAT", split='test')
```

As some questions are formatted as interleaved text and image(s), we recommend referring to the `./inference/eval_API.py` file for the correct way to query the LMM.

**File Structure**

* `./config/constants.py`: You can tweak the experiment settings here.
* `./config/prompts.py`: You can find the VQA prompt (both the CoT version and the non-CoT version) and the evaluation prompt.
* `./inference/eval_API.py`: The QA process, including how the LMM query context is structured and details about API calls, is defined in this file.
* `./eval.py`: The evaluation process for each question, including the QA phase and the answer evaluation phase.
* `./main.py`: The script in `main.py` loops over all questions and uses multithreading to speed up the evaluation process. Logging is taken care of in `main.py` as well.
* `./analyze.py`: Used to generate the leaderboard based on the logs. The leaderboard can be found under the directory `./analytics/`.

**Column Description**

- `index`: The index of the question in the dataset.
- `question`: The question text.
- `choices`: A list of the answer choices. Can be empty.
- `images`: The list of PIL images.
- `outside_knowledge_text`: The essential information for answering the question. Optional.
- `outside_knowledge_images`: The list of PIL images that are essential for answering the question. Can be empty.
- `answer`: The correct answer.
- `capability`: The VL capabilities required to answer the question. A list of strings.
- `human_cot`: The human annotation for the CoT reasoning process.

## Future Work

Going forward, we intend to further increase the diversity of the tasks in **MOAT**, involving more capability combinations and encompassing more domains and scenarios. Stay tuned!
