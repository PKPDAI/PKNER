# PKNER

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2024.02.12.580001-red)](https://www.biorxiv.org/content/10.1101/2024.02.12.580001v1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4646970.svg)](https://doi.org/10.5281/zenodo.4646970)

[**PKNER**](#pkner)| [**Setup**](#setup) | [**Download data and models**](#download-data-and-checkpoints) | [**Inference**](#inference) | [**Citing**](#citation)

This repository contains code to perform Named Entity Recognition of Pharmacokinetic Parameters in the scientific literature.

<p align="center">
<img src="nerdemo.gif" style="width: auto; height: 350px;"/>
</p>

## Setup

1. Create and activate a virtual environment with `python 3.8.12` installed

2. Install this repo to get started:

````bash
git clone https://github.com/PKPDAI/PKNER
cd PKNER
pip install -e .
````

## Download data and checkpoints

````bash
sh scripts/download_annotations.sh
sh scripts/download_pretrained_biobert_pkner.sh
````

## Inference
### Trained spaCy inference

To use NER for PK parameters with spaCy make sure scispaCy is installed (`pip install scispacy`). Then install the NER package for PK parameters through:

````
pip install https://pkannotations.blob.core.windows.net/nerdata/trained_models/en_pk_ner-0.0.0.tar.gz
````

You can use the model through:

```python
import spacy

nlp = spacy.load("en_pk_ner")
doc = nlp("Parameter estimations for a subject of 34kg indicated values of midazolam clearance of 34.7l·h-1, a central volume of distribution of 27.9l and a peripheral volume of distribution of 413l.")
for ent in doc.ents:
    print(ent)
#>>> clearance
#>>> central volume of distibution
#>>> peripheral volume of distribution
```

### Inference and evaluation with PKNER BERT-based models

```shell
python scripts/evaluate_bert.py \
   --model-checkpoint checkpoints/biobert-ner-trained.ckpt \
   --predict-file-path data/test.jsonl \
   --display-errors \
   --batch-size 256 \
   --gpu \
   --n-workers 12
```

## Citation

```bibtex
@article{hernandez2024named,
  title={Named Entity Recognition of Pharmacokinetic parameters in the scientific literature},
  author={Hernandez, Ferran Gonzalez and Nguyen, Quang and Smith, Victoria C and Cordero, Jose Antonio and Ballester, Maria Rosa and Duran, Marius and Sole, Albert and Chotsiri, Palang and Wattanakul, Thanaporn and Mundin, Gill and others},
  journal={bioRxiv},
  pages={2024--02},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
