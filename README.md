# Quality Estimation for Conversational Data

This repository contains the code for our paper: [Is Context Helpful for Chat Translation Evaluation?](https://arxiv.org/pdf/2403.08314)

## Abstract
Despite the recent success of automatic metrics for assessing translation quality, their application in evaluating the quality of machine-translated chats has been limited. Unlike more structured texts like news, chat conversations are often unstructured, short, and heavily reliant on contextual information. This poses questions about the reliability of existing sentence-level metrics in this domain as well as the role of context in assessing the translation quality. Motivated by this, we conduct a meta-evaluation of existing sentence-level automatic metrics, primarily designed for structured domains such as news, to assess the quality of machine-translated chats. We find that reference-free metrics lag behind reference-based ones, especially when evaluating translation quality in out-of-English settings. We then investigate how incorporating conversational contextual information in these metrics affects their performance. Our findings show that augmenting neural learned metrics with contextual information helps improve correlation with human judgments in the reference-free scenario and when evaluating translations in out-of-English settings. Finally, we propose a new evaluation metric, Context-MQM, that utilizes bilingual context with a large language model (LLM) and further validate that adding context helps even for LLM-based evaluation metrics.

## Installation

1. Install COMET from source. This enables the document level extension of COMET as detailed [here](https://statmt.org/wmt22/pdf/2022.wmt-1.6.pdf).

```
    pip install git+https://github.com/Unbabel/COMET.git
```

2. Install [bleurt](https://arxiv.org/abs/2004.04696) and download BLEURT checkpoint.
```
    pip install git+https://github.com/google-research/bleurt.git
    wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
    unzip BLEURT-20.zip
```

3. Install MetricX from [here](https://github.com/google-research/metricx).

4. Install other requirements using requirements.txt

```
    pip install -r requirements.txt
```

## Metrics Benchmark & Contextual Comet QE

To reproduce the results from Tables 2 and 3 from our paper, run:

```
    bash run_all_paper_eval.sh
```
