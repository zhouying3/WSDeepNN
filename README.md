# WSDeepNN
A Weight-Selection Strategy for training Deep Neural Networks for Imbalanced Classification

## Motivation
Deep Neural Networks (DNN) have recently received great attention due to their superior performance in many machining-learning problems. However, the use of DNN is still impeded, if the input data is imbalanced. Imbalanced classification refers to the problem that one class contains a much smaller number of samples than the others in classification. It poses a great challenge to existing classifiers including DNN, due to the difficulty in recognizing the minority class. So far, there are still limited studies on how to train DNN for imbalanced classification. 

## Results
In this study, we propose a new strategy to reduce over-fitting in training DNN for imbalanced classification based on weight selection. In training DNN, by splitting the original training set into two subsets, one used for training to update weights, and the other for validation to select weights, the weights that render the best performance in the validation set would be selected. To our knowledge, it is the first systematic study to examine a weight-selection strategy on training DNN for imbalanced classification. Demonstrated by experiments on 10 imbalanced datasets obtained from MNIST, the DNN trained by our new strategy outperformed the DNN trained by a standard strategy and the DNN trained by cost-sensitive learning with statistical significance (p=0.00512). Surprisingly, the DNN trained by our new strategy was trained on 20% less training images, corresponding to 12,000 less training images, but still achieved an outperforming performance in all 10 imbalanced datasets.

## Proposed Strategy
<p align="center"><img width=60% src="https://github.com/antoniosehk/WSDeepNN/blob/master/content/overview.png"></p>
DNN with 3 hidden layers were trained to classify if a digit image pertains to the -ve class (minority) or +ve class (majority) on 10 imbalanced datasets obtained from MNIST. b) An illustration of the standard strategy, where the DNN is trained on the entire training set to update weights. c) An illustration of the proposed strategy, where the original training set is split into two subsets, one for training to update weights and the other for weight selection. The weights which obtain the best performance on the validation set would be selected.
