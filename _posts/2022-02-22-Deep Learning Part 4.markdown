---
layout: post
title: Deep Learning Part 4
date: 2022-02-22
categories: data_science
---

# Getting started with neural networks: Classification and regression

This part covers:
- First examples of real-world machine learning workflows
- Handling classification problems over vector data
- Handling continuous regression problems over vector data


This will be the fourth part of a series of posts for my own reference and continued professional development in deep learning. It should mostly follow important points taken from François Chollet's book Deep Learning with Python, Second Edition. 

The main book itself can be found at [Manning.com](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-1/)


```python
# conda create --name deep-learning
# conda activate deep-learning
# pip install tensorflow==2.6 keras==2.6
```

In this part, we'll be conducting three common use cases of neural networks:

1. Binary classification of movie reviews as positive or negative
2. Multi-class classification by classifying news wires by topic
3. Scalar data Estimating price of a house, given real-estate data

## Classifying movie reviews: A binary classification example

### The IMDB dataset

**Loading the IMDB dataset**


```python
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    17465344/17464789 [==============================] - 0s 0us/step
    17473536/17464789 [==============================] - 0s 0us/step
    


```python
train_data[0]
```




    [1,
     14,
     22,
     16,
     43,
     530,
     973,
     1622,
     1385,
     65,
     458,
     4468,
     66,
     3941,
     4,
     173,
     36,
     256,
     5,
     25,
     100,
     43,
     838,
     112,
     50,
     670,
     2,
     9,
     35,
     480,
     284,
     5,
     150,
     4,
     172,
     112,
     167,
     2,
     336,
     385,
     39,
     4,
     172,
     4536,
     1111,
     17,
     546,
     38,
     13,
     447,
     4,
     192,
     50,
     16,
     6,
     147,
     2025,
     19,
     14,
     22,
     4,
     1920,
     4613,
     469,
     4,
     22,
     71,
     87,
     12,
     16,
     43,
     530,
     38,
     76,
     15,
     13,
     1247,
     4,
     22,
     17,
     515,
     17,
     12,
     16,
     626,
     18,
     2,
     5,
     62,
     386,
     12,
     8,
     316,
     8,
     106,
     5,
     4,
     2223,
     5244,
     16,
     480,
     66,
     3785,
     33,
     4,
     130,
     12,
     16,
     38,
     619,
     5,
     25,
     124,
     51,
     36,
     135,
     48,
     25,
     1415,
     33,
     6,
     22,
     12,
     215,
     28,
     77,
     52,
     5,
     14,
     407,
     16,
     82,
     2,
     8,
     4,
     107,
     117,
     5952,
     15,
     256,
     4,
     2,
     7,
     3766,
     5,
     723,
     36,
     71,
     43,
     530,
     476,
     26,
     400,
     317,
     46,
     7,
     4,
     2,
     1029,
     13,
     104,
     88,
     4,
     381,
     15,
     297,
     98,
     32,
     2071,
     56,
     26,
     141,
     6,
     194,
     7486,
     18,
     4,
     226,
     22,
     21,
     134,
     476,
     26,
     480,
     5,
     144,
     30,
     5535,
     18,
     51,
     36,
     28,
     224,
     92,
     25,
     104,
     4,
     226,
     65,
     16,
     38,
     1334,
     88,
     12,
     16,
     283,
     5,
     16,
     4472,
     113,
     103,
     32,
     15,
     16,
     5345,
     19,
     178,
     32]




```python
train_labels[0]
```




    1




```python
max([max(sequence) for sequence in train_data])
```




    9999



**Decoding reviews back to text**


```python
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
    1646592/1641221 [==============================] - 0s 0us/step
    1654784/1641221 [==============================] - 0s 0us/step
    

### Preparing the data

**Encoding the integer sequences via multi-hot encoding**


```python
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
```


```python
x_train[0]
```




    array([0., 1., 1., ..., 0., 0., 0.])




```python
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
```

### Building your model

**Model definition**


```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
```

**Compiling the model**


```python
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
```

### Validating your approach

**Setting aside a validation set**


```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

**Training your model**


```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```

    Epoch 1/20
    30/30 [==============================] - 1s 21ms/step - loss: 0.5129 - accuracy: 0.7872 - val_loss: 0.3934 - val_accuracy: 0.8648
    Epoch 2/20
    30/30 [==============================] - 0s 11ms/step - loss: 0.3028 - accuracy: 0.9073 - val_loss: 0.3082 - val_accuracy: 0.8852
    Epoch 3/20
    30/30 [==============================] - 0s 11ms/step - loss: 0.2211 - accuracy: 0.9309 - val_loss: 0.2849 - val_accuracy: 0.8881
    Epoch 4/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.1715 - accuracy: 0.9461 - val_loss: 0.2820 - val_accuracy: 0.8865
    Epoch 5/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.1411 - accuracy: 0.9568 - val_loss: 0.2872 - val_accuracy: 0.8847
    Epoch 6/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.1163 - accuracy: 0.9655 - val_loss: 0.3312 - val_accuracy: 0.8720
    Epoch 7/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0975 - accuracy: 0.9723 - val_loss: 0.3106 - val_accuracy: 0.8809
    Epoch 8/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0783 - accuracy: 0.9799 - val_loss: 0.3296 - val_accuracy: 0.8813
    Epoch 9/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0670 - accuracy: 0.9819 - val_loss: 0.3512 - val_accuracy: 0.8788
    Epoch 10/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0526 - accuracy: 0.9883 - val_loss: 0.4329 - val_accuracy: 0.8651
    Epoch 11/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0431 - accuracy: 0.9905 - val_loss: 0.4197 - val_accuracy: 0.8709
    Epoch 12/20
    30/30 [==============================] - 0s 9ms/step - loss: 0.0367 - accuracy: 0.9922 - val_loss: 0.4284 - val_accuracy: 0.8753
    Epoch 13/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0280 - accuracy: 0.9944 - val_loss: 0.4551 - val_accuracy: 0.8732
    Epoch 14/20
    30/30 [==============================] - 0s 11ms/step - loss: 0.0225 - accuracy: 0.9960 - val_loss: 0.4877 - val_accuracy: 0.8720
    Epoch 15/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0192 - accuracy: 0.9971 - val_loss: 0.5236 - val_accuracy: 0.8681
    Epoch 16/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0167 - accuracy: 0.9967 - val_loss: 0.5690 - val_accuracy: 0.8663
    Epoch 17/20
    30/30 [==============================] - 0s 10ms/step - loss: 0.0085 - accuracy: 0.9997 - val_loss: 0.5844 - val_accuracy: 0.8676
    Epoch 18/20
    30/30 [==============================] - 0s 9ms/step - loss: 0.0104 - accuracy: 0.9985 - val_loss: 0.6102 - val_accuracy: 0.8672
    Epoch 19/20
    30/30 [==============================] - 0s 9ms/step - loss: 0.0099 - accuracy: 0.9984 - val_loss: 0.6340 - val_accuracy: 0.8670
    Epoch 20/20
    30/30 [==============================] - 0s 9ms/step - loss: 0.0041 - accuracy: 0.9999 - val_loss: 0.6845 - val_accuracy: 0.8645
    


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])



**Plotting the training and validation loss**


```python
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_29_0.png">
    


**Plotting the training and validation accuracy**


```python
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_31_0.png">
    


**Retraining a model from scratch**


```python
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
```

    Epoch 1/4
    49/49 [==============================] - 1s 7ms/step - loss: 0.4696 - accuracy: 0.8082
    Epoch 2/4
    49/49 [==============================] - 0s 6ms/step - loss: 0.2698 - accuracy: 0.9091
    Epoch 3/4
    49/49 [==============================] - 0s 6ms/step - loss: 0.2048 - accuracy: 0.9281
    Epoch 4/4
    49/49 [==============================] - 0s 6ms/step - loss: 0.1723 - accuracy: 0.9397
    782/782 [==============================] - 1s 785us/step - loss: 0.2961 - accuracy: 0.8832
    


```python
results
```




    [0.29610514640808105, 0.8831599950790405]



### Using a trained model to generate predictions on new data


```python
model.predict(x_test)
```




    array([[0.2507104 ],
           [0.9987034 ],
           [0.9308133 ],
           ...,
           [0.1539098 ],
           [0.10201451],
           [0.67591244]], dtype=float32)



### Further experiments

### Wrapping up

## Classifying newswires: A multiclass classification example

### The Reuters dataset

**Loading the Reuters dataset**


```python
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters.npz
    2113536/2110848 [==============================] - 0s 0us/step
    2121728/2110848 [==============================] - 0s 0us/step
    


```python
len(train_data)
```




    8982




```python
len(test_data)
```




    2246




```python
train_data[10]
```




    [1,
     245,
     273,
     207,
     156,
     53,
     74,
     160,
     26,
     14,
     46,
     296,
     26,
     39,
     74,
     2979,
     3554,
     14,
     46,
     4689,
     4329,
     86,
     61,
     3499,
     4795,
     14,
     61,
     451,
     4329,
     17,
     12]



**Decoding newswires back to text**


```python
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in
    train_data[0]])
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters_word_index.json
    557056/550378 [==============================] - 0s 0us/step
    565248/550378 [==============================] - 0s 0us/step
    


```python
train_labels[10]
```




    3



### Preparing the data

**Encoding the input data**


```python
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
```

**Encoding the labels**


```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)
```


```python
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
```

### Building your model

**Model definition**


```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])
```

**Compiling the model**


```python
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```

### Validating your approach

**Setting aside a validation set**


```python
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]
```

**Training the model**


```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```

    Epoch 1/20
    16/16 [==============================] - 1s 19ms/step - loss: 2.6662 - accuracy: 0.5324 - val_loss: 1.7438 - val_accuracy: 0.6370
    Epoch 2/20
    16/16 [==============================] - 0s 12ms/step - loss: 1.4232 - accuracy: 0.7018 - val_loss: 1.2964 - val_accuracy: 0.7120
    Epoch 3/20
    16/16 [==============================] - 0s 12ms/step - loss: 1.0526 - accuracy: 0.7709 - val_loss: 1.1384 - val_accuracy: 0.7510
    Epoch 4/20
    16/16 [==============================] - 0s 11ms/step - loss: 0.8347 - accuracy: 0.8130 - val_loss: 1.0374 - val_accuracy: 0.7720
    Epoch 5/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.6684 - accuracy: 0.8611 - val_loss: 0.9920 - val_accuracy: 0.7790
    Epoch 6/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.5378 - accuracy: 0.8867 - val_loss: 0.9329 - val_accuracy: 0.8050
    Epoch 7/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.4284 - accuracy: 0.9092 - val_loss: 0.9189 - val_accuracy: 0.8070
    Epoch 8/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.3465 - accuracy: 0.9258 - val_loss: 0.9345 - val_accuracy: 0.8110
    Epoch 9/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.2874 - accuracy: 0.9341 - val_loss: 0.9038 - val_accuracy: 0.8160
    Epoch 10/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.2415 - accuracy: 0.9451 - val_loss: 0.9177 - val_accuracy: 0.8100
    Epoch 11/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.2096 - accuracy: 0.9489 - val_loss: 0.9184 - val_accuracy: 0.8160
    Epoch 12/20
    16/16 [==============================] - 0s 11ms/step - loss: 0.1838 - accuracy: 0.9523 - val_loss: 0.9605 - val_accuracy: 0.8090
    Epoch 13/20
    16/16 [==============================] - 0s 13ms/step - loss: 0.1626 - accuracy: 0.9545 - val_loss: 0.9768 - val_accuracy: 0.8170
    Epoch 14/20
    16/16 [==============================] - 0s 13ms/step - loss: 0.1518 - accuracy: 0.9551 - val_loss: 1.0273 - val_accuracy: 0.8000
    Epoch 15/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.1387 - accuracy: 0.9568 - val_loss: 1.0438 - val_accuracy: 0.8170
    Epoch 16/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.1312 - accuracy: 0.9578 - val_loss: 1.0379 - val_accuracy: 0.8040
    Epoch 17/20
    16/16 [==============================] - 0s 14ms/step - loss: 0.1247 - accuracy: 0.9551 - val_loss: 1.0351 - val_accuracy: 0.8190
    Epoch 18/20
    16/16 [==============================] - 0s 12ms/step - loss: 0.1174 - accuracy: 0.9583 - val_loss: 1.0857 - val_accuracy: 0.8070
    Epoch 19/20
    16/16 [==============================] - 0s 11ms/step - loss: 0.1165 - accuracy: 0.9564 - val_loss: 1.0859 - val_accuracy: 0.8060
    Epoch 20/20
    16/16 [==============================] - 0s 11ms/step - loss: 0.1131 - accuracy: 0.9577 - val_loss: 1.1245 - val_accuracy: 0.7990
    

**Plotting the training and validation loss**


```python
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_66_0.png">
    


**Plotting the training and validation accuracy**


```python
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_68_0.png">
    


**Retraining a model from scratch**


```python
model = keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)
results = model.evaluate(x_test, y_test)
```

    Epoch 1/9
    18/18 [==============================] - 0s 10ms/step - loss: 2.4749 - accuracy: 0.5518
    Epoch 2/9
    18/18 [==============================] - 0s 10ms/step - loss: 1.3496 - accuracy: 0.7141
    Epoch 3/9
    18/18 [==============================] - 0s 11ms/step - loss: 1.0149 - accuracy: 0.7821
    Epoch 4/9
    18/18 [==============================] - 0s 11ms/step - loss: 0.7977 - accuracy: 0.8323
    Epoch 5/9
    18/18 [==============================] - 0s 11ms/step - loss: 0.6341 - accuracy: 0.8687
    Epoch 6/9
    18/18 [==============================] - 0s 10ms/step - loss: 0.5043 - accuracy: 0.8947
    Epoch 7/9
    18/18 [==============================] - 0s 10ms/step - loss: 0.4008 - accuracy: 0.9168
    Epoch 8/9
    18/18 [==============================] - 0s 10ms/step - loss: 0.3245 - accuracy: 0.9299
    Epoch 9/9
    18/18 [==============================] - 0s 10ms/step - loss: 0.2748 - accuracy: 0.9393
    71/71 [==============================] - 0s 1ms/step - loss: 0.9611 - accuracy: 0.7907
    


```python
results
```




    [0.9610627293586731, 0.790739119052887]




```python
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
hits_array.mean()
```




    0.19412288512911843



### Generating predictions on new data


```python
predictions = model.predict(x_test)
```


```python
predictions[0].shape
```




    (46,)




```python
np.sum(predictions[0])
```




    0.99999994




```python
np.argmax(predictions[0])
```




    4



### A different way to handle the labels and the loss


```python
y_train = np.array(train_labels)
y_test = np.array(test_labels)
```


```python
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

### The importance of having sufficiently large intermediate layers

**A model with an information bottleneck**


```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))
```

    Epoch 1/20
    63/63 [==============================] - 1s 9ms/step - loss: 3.0133 - accuracy: 0.2476 - val_loss: 2.2469 - val_accuracy: 0.5560
    Epoch 2/20
    63/63 [==============================] - 0s 7ms/step - loss: 1.8257 - accuracy: 0.5673 - val_loss: 1.6499 - val_accuracy: 0.5860
    Epoch 3/20
    63/63 [==============================] - 0s 7ms/step - loss: 1.4868 - accuracy: 0.5900 - val_loss: 1.5347 - val_accuracy: 0.5890
    Epoch 4/20
    63/63 [==============================] - 0s 6ms/step - loss: 1.3241 - accuracy: 0.6185 - val_loss: 1.4505 - val_accuracy: 0.6190
    Epoch 5/20
    63/63 [==============================] - 0s 6ms/step - loss: 1.1768 - accuracy: 0.6723 - val_loss: 1.3833 - val_accuracy: 0.6640
    Epoch 6/20
    63/63 [==============================] - 0s 6ms/step - loss: 1.0630 - accuracy: 0.7181 - val_loss: 1.3427 - val_accuracy: 0.6800
    Epoch 7/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.9696 - accuracy: 0.7440 - val_loss: 1.3498 - val_accuracy: 0.6960
    Epoch 8/20
    63/63 [==============================] - 0s 7ms/step - loss: 0.8890 - accuracy: 0.7674 - val_loss: 1.3341 - val_accuracy: 0.6940
    Epoch 9/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.8193 - accuracy: 0.7848 - val_loss: 1.3430 - val_accuracy: 0.7040
    Epoch 10/20
    63/63 [==============================] - 0s 7ms/step - loss: 0.7587 - accuracy: 0.8003 - val_loss: 1.3472 - val_accuracy: 0.7130
    Epoch 11/20
    63/63 [==============================] - 0s 7ms/step - loss: 0.7093 - accuracy: 0.8113 - val_loss: 1.3533 - val_accuracy: 0.7120
    Epoch 12/20
    63/63 [==============================] - 0s 7ms/step - loss: 0.6672 - accuracy: 0.8183 - val_loss: 1.4235 - val_accuracy: 0.7140
    Epoch 13/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.6305 - accuracy: 0.8242 - val_loss: 1.4464 - val_accuracy: 0.7180
    Epoch 14/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.5993 - accuracy: 0.8301 - val_loss: 1.4878 - val_accuracy: 0.7130
    Epoch 15/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.5712 - accuracy: 0.8364 - val_loss: 1.5241 - val_accuracy: 0.7110
    Epoch 16/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.5472 - accuracy: 0.8388 - val_loss: 1.5877 - val_accuracy: 0.7170
    Epoch 17/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.5246 - accuracy: 0.8493 - val_loss: 1.6331 - val_accuracy: 0.7100
    Epoch 18/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.5018 - accuracy: 0.8555 - val_loss: 1.6974 - val_accuracy: 0.7060
    Epoch 19/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.4815 - accuracy: 0.8636 - val_loss: 1.6944 - val_accuracy: 0.7150
    Epoch 20/20
    63/63 [==============================] - 0s 6ms/step - loss: 0.4654 - accuracy: 0.8701 - val_loss: 1.7913 - val_accuracy: 0.7060
    




    <keras.callbacks.History at 0x200332bd9d0>



### Further experiments

### Wrapping up

## Predicting house prices: A regression example

### The Boston Housing Price dataset

**Loading the Boston housing dataset**


```python
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
    57344/57026 [==============================] - 0s 0us/step
    65536/57026 [==================================] - 0s 0us/step
    


```python
train_data.shape
```




    (404, 13)




```python
test_data.shape
```




    (102, 13)




```python
train_targets
```




    array([15.2, 42.3, 50. , 21.1, 17.7, 18.5, 11.3, 15.6, 15.6, 14.4, 12.1,
           17.9, 23.1, 19.9, 15.7,  8.8, 50. , 22.5, 24.1, 27.5, 10.9, 30.8,
           32.9, 24. , 18.5, 13.3, 22.9, 34.7, 16.6, 17.5, 22.3, 16.1, 14.9,
           23.1, 34.9, 25. , 13.9, 13.1, 20.4, 20. , 15.2, 24.7, 22.2, 16.7,
           12.7, 15.6, 18.4, 21. , 30.1, 15.1, 18.7,  9.6, 31.5, 24.8, 19.1,
           22. , 14.5, 11. , 32. , 29.4, 20.3, 24.4, 14.6, 19.5, 14.1, 14.3,
           15.6, 10.5,  6.3, 19.3, 19.3, 13.4, 36.4, 17.8, 13.5, 16.5,  8.3,
           14.3, 16. , 13.4, 28.6, 43.5, 20.2, 22. , 23. , 20.7, 12.5, 48.5,
           14.6, 13.4, 23.7, 50. , 21.7, 39.8, 38.7, 22.2, 34.9, 22.5, 31.1,
           28.7, 46. , 41.7, 21. , 26.6, 15. , 24.4, 13.3, 21.2, 11.7, 21.7,
           19.4, 50. , 22.8, 19.7, 24.7, 36.2, 14.2, 18.9, 18.3, 20.6, 24.6,
           18.2,  8.7, 44. , 10.4, 13.2, 21.2, 37. , 30.7, 22.9, 20. , 19.3,
           31.7, 32. , 23.1, 18.8, 10.9, 50. , 19.6,  5. , 14.4, 19.8, 13.8,
           19.6, 23.9, 24.5, 25. , 19.9, 17.2, 24.6, 13.5, 26.6, 21.4, 11.9,
           22.6, 19.6,  8.5, 23.7, 23.1, 22.4, 20.5, 23.6, 18.4, 35.2, 23.1,
           27.9, 20.6, 23.7, 28. , 13.6, 27.1, 23.6, 20.6, 18.2, 21.7, 17.1,
            8.4, 25.3, 13.8, 22.2, 18.4, 20.7, 31.6, 30.5, 20.3,  8.8, 19.2,
           19.4, 23.1, 23. , 14.8, 48.8, 22.6, 33.4, 21.1, 13.6, 32.2, 13.1,
           23.4, 18.9, 23.9, 11.8, 23.3, 22.8, 19.6, 16.7, 13.4, 22.2, 20.4,
           21.8, 26.4, 14.9, 24.1, 23.8, 12.3, 29.1, 21. , 19.5, 23.3, 23.8,
           17.8, 11.5, 21.7, 19.9, 25. , 33.4, 28.5, 21.4, 24.3, 27.5, 33.1,
           16.2, 23.3, 48.3, 22.9, 22.8, 13.1, 12.7, 22.6, 15. , 15.3, 10.5,
           24. , 18.5, 21.7, 19.5, 33.2, 23.2,  5. , 19.1, 12.7, 22.3, 10.2,
           13.9, 16.3, 17. , 20.1, 29.9, 17.2, 37.3, 45.4, 17.8, 23.2, 29. ,
           22. , 18. , 17.4, 34.6, 20.1, 25. , 15.6, 24.8, 28.2, 21.2, 21.4,
           23.8, 31. , 26.2, 17.4, 37.9, 17.5, 20. ,  8.3, 23.9,  8.4, 13.8,
            7.2, 11.7, 17.1, 21.6, 50. , 16.1, 20.4, 20.6, 21.4, 20.6, 36.5,
            8.5, 24.8, 10.8, 21.9, 17.3, 18.9, 36.2, 14.9, 18.2, 33.3, 21.8,
           19.7, 31.6, 24.8, 19.4, 22.8,  7.5, 44.8, 16.8, 18.7, 50. , 50. ,
           19.5, 20.1, 50. , 17.2, 20.8, 19.3, 41.3, 20.4, 20.5, 13.8, 16.5,
           23.9, 20.6, 31.5, 23.3, 16.8, 14. , 33.8, 36.1, 12.8, 18.3, 18.7,
           19.1, 29. , 30.1, 50. , 50. , 22. , 11.9, 37.6, 50. , 22.7, 20.8,
           23.5, 27.9, 50. , 19.3, 23.9, 22.6, 15.2, 21.7, 19.2, 43.8, 20.3,
           33.2, 19.9, 22.5, 32.7, 22. , 17.1, 19. , 15. , 16.1, 25.1, 23.7,
           28.7, 37.2, 22.6, 16.4, 25. , 29.8, 22.1, 17.4, 18.1, 30.3, 17.5,
           24.7, 12.6, 26.5, 28.7, 13.3, 10.4, 24.4, 23. , 20. , 17.8,  7. ,
           11.8, 24.4, 13.8, 19.4, 25.2, 19.4, 19.4, 29.1])



### Preparing the data

**Normalizing the data**


```python
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
```

### Building your model

**Model definition**


```python
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model
```

### Validating your approach using K-fold validation

**K-fold validation**


```python
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
```

    Processing fold #0
    Processing fold #1
    Processing fold #2
    Processing fold #3
    


```python
all_scores
```




    [1.9171539545059204, 2.4800946712493896, 2.3863155841827393, 2.3695228099823]




```python
np.mean(all_scores)
```




    2.2882717549800873



**Saving the validation logs at each fold**


```python
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)
```

    Processing fold #0
    Processing fold #1
    Processing fold #2
    Processing fold #3
    

**Building the history of successive mean K-fold validation scores**


```python
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
```

**Plotting validation scores**


```python
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_109_0.png">
    


**Plotting validation scores, excluding the first 10 data points**


```python
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
```


    
<img src="/assets/images/DLwPCh4/output_111_0.png">
    


**Training the final model**


```python
model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
```

    4/4 [==============================] - 0s 667us/step - loss: 11.8174 - mae: 2.3855
    


```python
test_mae_score
```




    2.3855087757110596



### Generating predictions on new data


```python
predictions = model.predict(test_data)
predictions[0]
```




    array([7.513255], dtype=float32)



### Wrapping up

## Summary
