# Aspect-to Scope Multi-view Contrastive Learning (A2SMvCL)

This repository contains Pytorch implementation for "[Aspect-to-Scope Oriented Multi-view Contrastive Learning for Aspect-based Sentiment Analysis](https://aclanthology.org/2023.findings-emnlp.727/)" (Findings of EMNLP 2023)

## Requirements
```
python = 3.7
pytorch = 1.4
transformers = 3.2.0 
```

## Get Start
1. Prepare data
   
   We  provide the parsed data at directory **dataset**

2. Training
   
   ```
   sh run.sh
   ```

## Citation
**Please kindly cite our paper if this paper and the code are helpful.**
```
@inproceedings{chai-etal-2023-aspect,
    title = "Aspect-to-Scope Oriented Multi-view Contrastive Learning for Aspect-based Sentiment Analysis",
    author = "Chai, Heyan  and
      Yao, Ziyi  and
      Tang, Siyu  and
      Wang, Ye  and
      Nie, Liqiang  and
      Fang, Binxing  and
      Liao, Qing",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.727",
    doi = "10.18653/v1/2023.findings-emnlp.727",
    pages = "10902--10913",
    abstract = "Aspect-based sentiment analysis (ABSA) aims to align aspects and corresponding sentiment expressions, so as to identify the sentiment polarities of specific aspects. Most existing ABSA methods focus on mining syntactic or semantic information, which still suffers from noisy interference introduced by the attention mechanism and dependency tree when multiple aspects exist in a sentence. To address these issues, in this paper, we revisit ABSA from a novel perspective by proposing a novel scope-assisted multi-view graph contrastive learning framework. It not only mitigates noisy interference for better locating aspect and its corresponding sentiment opinion with aspect-specific scope, but also captures the correlation and difference between sentiment polarities and syntactic/semantic information. Extensive experiments on five benchmark datasets show that our proposed approach substantially outperforms state-of-the-art methods and verifies the effectiveness and robustness of our model.",
}


```
