# Mixture-of-Experts-Sentence-Similarity
 
This repository serves as the official code base for the paper _Contrastive Learning and Mixture of Experts Enables Precise Vector Embeddings_

Logan Hallee, Rohan Kapur, Arjun Patel, Jason P. Gleghorn, and Bohdan Khomtchouk

Preprint: [Contrastive Learning and Mixture of Experts Enables Precise Vector Embeddings](https://arxiv.org/abs/2401.15713)

Peer review: _preparing submission_

## Data and models
[Huggingface](https://huggingface.co/collections/GleghornLab/sentence-similarity-663d8679468f3aaf46619c35)

## Main findings
* Extending BERT models with N experts copied from their MLP section is highly effective for fine-tuning on downstream tasks, including multitask or multidomain data.
* N experts are almost as effective as N individual models trained on N domains for sentence similarity tasks, we have tried up to five.
* Enforced routing of experts can be handled with added special tokens for sentence-wise routing or designed token type IDs for token-wise routing, even when the router is a single linear layer. Enforced routing can also be accomplished by passing a list of desired indices. Mutual information based loss with top-k outputs also works well to correlate expert activation with specific types of data.
* Cocitation networks are highly effective for gathering similar niche papers.
* Using dot product with a learned temperature may be a more effective contrastive loss than standard Multiple Negatives Ranking loss.
### Surprising findings
* Small BERT models are not more effective with N experts, perhaps due to small shared attention layers. Our data supports that this threshold may be roughly 100 million parameters.
* MoE extension with a SINGLE MoE layer gets 85% of the full MoE extension on average.
* Token-wise routing is better than sentence-wise routing in general, even for sentence-wise tasks.

## Applications of this work
* Better vector databases / retrieval augmentation
* Extending any sufficiently large BERT model with N experts for N tasks.
* Vocabulary extension of BERT models with N experts for N vocabularies.

## [Docs](https://github.com/Gleghorn-Lab/Mixture-of-Experts-Sentence-Similarity/tree/main/documentation)
Coming soon

## Please cite
```
@article{hallee2024contrastive,
      title={Contrastive Learning and Mixture of Experts Enables Precise Vector Embeddings}, 
      author={Logan Hallee and Rohan Kapur and Arjun Patel and Jason P. Gleghorn and Bohdan Khomtchouk},
      year={2024},
      eprint={2401.15713},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
and upvote on [Huggingface](https://huggingface.co/papers/2401.15713)!
