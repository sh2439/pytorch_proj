# Word Analogy and Debiasing
---
This notebook contains word analogy, debiasing and equalizing taks. With the help of modern word embbeddings (e.g. GloVe, word2vec), we are able to make use of word vectors and accomplish these tasks.
1. **Word Analogy:** Compute word analogy. For example, 'China' is to 'Mandarin' as 'France' is to 'French'.
2. **Debiasing:** The dataset which was used to train the word embeddings can reflect the some bias of human language. Gender bias is a significant one. 
3. **Equalizing:** Some words are gender-specific. For example, we may assume gender is the only difference between 'girl' and 'boy'. Therefore, they should have the same distance from other dimensions.

### Acknowledgement:
Some ideas come from [Deep Learning Course on Coursera](https://www.deeplearning.ai/deep-learning-specialization/) (e.g., the debiasing and equalizing equations) and the [paper](https://arxiv.org/abs/1607.06520).
