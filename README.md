# Introduction

![](https://user-images.githubusercontent.com/111342294/203943452-34caa175-01f4-47e4-bf57-d49e39276b28.png)


We introduce an extra visual transformer as the alignment-ware image encoder and an extra text transformer as the alignment ware text encoder before multimodal fusion. We consider alignment in the following three aspects: 1) document-level alignment by leveraging the cross-modal and intra-modal contrastive loss; 2) global-local alignment for modeling localized and structural information in document images; and 3) local-level alignment for more accurate patch-level information. For more details, please refer to our paper:  **Alignment-Enriched Tuning for Patch-Level Pre-trained Document Image Models[pdf]**


# Finetuned Model

| Model | F1      | Link      |
|:--------:| :-------------:|:-------------:|
| FUNSD | 91.55 | [download]() |
| CORD | 97.04 |[download]() |
| RVL-CDIP |  |[download]() |


# Train

```
git clone 
```

### Finetune on FUNSD / CORD

```

```

### Finetune on RVL-CDIP

```

```


# Acknowledgements

We referenced the code of ALBEF and TCL when implementing AET in github, In this repository, we used three public benchmark datasets, FUNSD , CORD , RVL-CDIP.

# License

```

```
