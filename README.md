# MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding

This is the official GitHub repo of our paper *MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding* (INSERT ARXIV LINK HERE). All necessary data and code can be found here. Refer to our paper and website for more info.

![fig1](.\figures\fig1.png)

## Overview

**MOAT** (**M**ultimodal model **O**f **A**ll **T**rades) is a challenging benchmark for large multimodal models (LMMs). It consists of vision language (VL) tasks that require the LMM to integrate several VL capabilities and engage in human-like generalist visual problem solving. Moreover, many tasks in **MOAT** focus on LMMs' capability to ground complex text and visual instructions, which is crucial for the application of LMMs in-the-wild. Developing on the VL capability taxonomies proposed in previous benchmark papers, we define 10 fundamental VL capabilities in **MOAT**. 

![fig2](.\figures\fig2.png)

Notably, we purposefully insulated **MOAT** from the influence of domain knowledge, text generation style, and other external factors by making the questions close-ended (i.e. have a single short answer) and solvable with the information and hints provided in the question itself. This allows **MOAT** to focus on fundamental generalist VL capabilities. We also did not include VL capabilities like *general object recognition* and *attribute recognition* in our taxonomy, since these are required by all **MOAT** tasks, and performance on these fronts can be reflected in the overall accuracy on **MOAT**.

## Benchmark Composition

**MOAT** tasks require LMMs to integrate up to 6 fundamental VL capabilities. We report the proportion of questions requiring each VL capability, the distribution of the number of VL capabilities required, and the 15 most common capability combinations required in **MOAT**.

<img src=".\figures\fig3a.png" alt="fig3a" style="zoom:50%;" /><img src=".\figures\fig3b.png" alt="fig3b" style="zoom:50%;" />

<img src=".\figures\fig3c.png" alt="fig3c" style="zoom:70%;" />

## Leaderboard

**ALL** existing LMMs, proprietary and open source, perform very poorly on **MOAT**, with the best performing model (OpenAI o1) achieving an accuracy (38.8%) less than half of that achieved by humans (82.7%). For individual VL capabilities, **CNT**, **RLA**, **3DTF** and **3DQNT** saw consistent poor performance by LMMs. In addition, **GNDT** and **GNDV** performance did not scale well with model size. Please refer to our paper for more detailed analysis of the results, as well as discussion on the implication of LMM architecture choices such as tiling and built-in CoT reasoning (or '*thinking*') capability.

![fig4](.\figures\fig4.png)

## Data and Code

**Quickstart**

* Dependencies are listed in `./requirements.txt`. We used `Python 3.12.8` in our experiments. Run `pip install -r requirements.txt` to install all dependencies. 

* The questions can be found in `./dataset/questions.json`. The images can be found in the various directories under `./dataset/`.

* To evaluate an LMM, run `python main.py`, and the results will be logged under `./logs/`. Change the model name, API endpoint, and API key in `./configs/constants.py`.

**Detailed Explanation**

* `./config/constants.py`: You can tweak the experiment settings here.
* `./config/prompts.py`: You can find the VQA prompt (both the CoT version and the non-CoT version) and the evaluation prompt.
* `./dataset/`: The questions are in `./dataset/questions.json`, while the images are in the various folders here.
* `./inference/eval_API.py`: The QA process, including how the LMM query context is structured and details about API calls, is defined in this file.
* `./eval.py`: The evaluation process for each question, including the QA phase and the answer evaluation phase.
* `./main.py`: The script in `main.py` loops over all questions and uses multithreading to speed up the evaluation process. Logging is taken care of in `main.py` as well.
* `./analyze.py`: Used to generate the leaderboard based on the logs. The leaderboard can be found under the directory `./analytics/`.

## Future Work

Going forward, we intend to further increase the diversity of the tasks in **MOAT**, involving more capability combinations and encompassing more domains and scenarios. Stay tuned!