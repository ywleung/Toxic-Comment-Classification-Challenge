This project contains IPyhton Notebook files for a Kaggle competition <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge">Toxic Comment Classification Challenge</a>.

The results should score around 0.98.

Conclusion:
1) GloVe 300D word embedding is used, but no significant improvement from GloVe 50D word embedding
2) Bi-LSTM with attention layer model performs a bit better than the model with single Bi-LSTM.
3) It's easy to overfit in Deep Learning model. Both model starts overfitting after the second epoch on training.

What can be done to get a better score:
1) play around with tokenization (emoji/ slang)
2) try to normalize tokens (stemming/ lemmatization)
3) try different word embedding (fastText) or concatenate GloVe and fastText embedding
4) try different model (capsule layer/ LSTM-CNN)
