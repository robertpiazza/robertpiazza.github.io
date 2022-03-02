---
layout: post
title: Glossary of ML Terms
date: 2022-02-26
categories: data_science
---

# Glossary of Machine Learning terms
-   _Sample_ or _input_—One data point that goes into your model.
    
-   _Prediction_ or _output_—What comes out of your model.
    
-   _Target_—The truth. What your model should ideally have predicted, according to an external source of data.
    
-   _Prediction error_ or _loss value_—A measure of the distance between your model’s prediction and the target.
    
-   _Classes_—A set of possible labels to choose from in a classification problem. For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.
    
-   _Label_ —A specific instance of a class annotation in a classification problem. For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.
    
-   _Ground-truth_ or _annotations_—All targets for a dataset, typically collected by humans.
    
-   _Binary classification_—A classification task where each input sample should be categorized into two exclusive categories.
    
-   _Multiclass classification_—A classification task where each input sample should be categorized into more than two categories: for instance, classifying handwritten digits.
    
-   _Multilabel classification_—A classification task where each input sample can be assigned multiple labels. For instance, a given image may contain both a cat and a dog and should be annotated both with the “cat” label and the “dog” label. The number of labels per image is usually variable.
    
-   _Scalar regression_—A task where the target is a continuous scalar value. Predicting house prices is a good example: the different target prices form a continuous space.
    
-   _Vector regression_—A task where the target is a set of continuous values: for example, a continuous vector. If you’re doing regression against multiple values (such as the coordinates of a bounding box in an image), then you’re doing vector regression.
    
-   _Mini-batch_ or _batch_—A small set of samples (typically between 8 and 128) that are processed simultaneously by the model. The number of samples is often a power of 2, to facilitate memory allocation on GPU. When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.

- _Optimization_ - The process of getting the best performance possible from the training data for a particular model. 