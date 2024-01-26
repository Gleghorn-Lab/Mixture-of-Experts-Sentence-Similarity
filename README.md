# Multi-Vocab-MOE
 
This repository serves as the code base for the paper _Contrastive Learning and Mixture of Experts Enables Precise Vector Embeddings_

Rohan Kapur*, Logan Hallee*, Arjun Patel, and Bohdan Bohdan Khomtchouk

<sub><sup>* equal contribution</sup></sub>

Preprint: doi

Peer review: doi

## Main findings
* Extending BERT models with N experts copied from their MLP section is highly effective for fine-tuning on downstream tasks, including multitask or multidomain data.
* N experts are exactly as effective as N individual models trained on N domains for sentence similarity tasks.
* Small BERT models are not more effective with N experts, likely due to small shared attention layers.
* Enforced routing of experts can be handled with added special tokens for sentence-wise routing or token type IDs for token-wise routing, even when the router is a single linear layer. Enforced routing can also be accomplished by passing a list of desired indices.
* Cocitation networks are highly effective for gathering similar niche papers.

## Code Details
* data_compilation

**citation_extraction.R**
Pulls all papers from PubMedCentral with abstracts that match inputted MeSH terms and timeframe.

**clean_datasets.py**
Takes datasets from citation_extraction.R and converts them into format used for cocitation analysis, including paper abstract, year, papers it cites, and papers cited by.

* **main.py**

Argparse to train or evaluate the models mentioned in the paper. run ```main.py -h``` in commmand line or ```%run main.py -h``` in jupyter for details.

* **model.py**

Contains Huggingface-inspired and adaptable versions of BERT with MOE capabilities, and new BertForSentenceSimilarity. Any Huggingface BERT model can be extended with N experts needed from the MLP sections of the models.

* **metrics.py**

All metrics used for training and evaluation.

* **losses.py**

All losses used in training, including various contrastive and router losses.

* **trainer.py**

Functions for PyTorch dataset compilation and training / evaluation of models, with versions for BERT or MOE-BERT.

## Please cite
bibtex
