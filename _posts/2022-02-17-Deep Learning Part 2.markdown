---
layout: post
title:  "Deep Learning Part 2"
date:   2022-02-17
categories: data_science
---
```python
# conda create --name deep-learning
# conda activate deep-learning
# pip install tensorflow==2.6 keras==2.6
```

This is a companion notebook for the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). The original notebook only contained runnable code blocks and section titles, and omits everything else. I, Robert Piazza, have added my own notes and comments to the book for greater learning and reference capabilities. 

# The mathematical building blocks of neural networks

> This chapter covers
> - A first example of a neural network
> - Tensors and tensor operations
> - How neural networks learn via backpropagation and gradient descent


- MNIST dataset (Modified National Institute of Standards and Technology) is the "Hello World" of neural networks and machine learning.

- Deep learning requires familiarity with:
    - tensors, 
    - tensor operations
    - differentiation
    - gradient descent
    
    
  


## A first look at a neural network

The MNIST dataset is a series of handwritten digits, digitized into 28x28 grayscale images and appropriately classified from 0 to 9. We're going to make a basic network to classify them. 

**Loading the MNIST dataset in Keras**


```python
from tensorflow.keras.datasets import mnist
import numpy as np

#standard syntax for keras datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images_2, train_labels_2), (test_images_2, test_labels_2) = mnist.load_data()

```


```python
#how many dimensions?
train_images.shape
```




    (60000, 28, 28)



There are **60,000*** training examples, each 28 by 28 pixesl


```python
len(train_labels)
```




    60000



Good data should have the same number of labels as training examples


```python
train_labels
```




    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)




```python
test_images.shape
```




    (10000, 28, 28)



Now we know we have 10K testing images. This is similar to a standard 80/20 or 90/10 test split but it's more like a 15% test split. 


```python
len(test_labels)
```




    10000




```python
test_labels
```




    array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)



**The network architecture**

Two dense layers (also called *fully connected*) will be the basis of the neural architecture. Their job is to extract more useful representations of the data into a distilled version of the information. This data distillation is both a means of extracting information from the input and a form of data compression for free!




```python
from tensorflow import keras
from tensorflow.keras import layers

#Very Simple initial architecture - 512 nodes on the first layer, 
#10 output nodes on the second layer.
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

**The compilation step**

Compilation requires three more steps:

- An optimizer
- A loss function
- Metrics to monitor during training and testing


```python
# A lot of options are possible here but this is pretty standard for this type of job. 
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

**Preparing the image data**

- Turn each image into a single vector to be shown to the network
- normalize each grayscale number from its 0-255 range (uint8 data type) to 0 to 1 (float32)



```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

**"Fitting" the model**

the [model].fit([parameters]) syntax is consistent throughout Keras and scikit-learn 


```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    469/469 [==============================] - 2s 3ms/step - loss: 0.2557 - accuracy: 0.9257
    Epoch 2/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.1030 - accuracy: 0.9694
    Epoch 3/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0675 - accuracy: 0.9796
    Epoch 4/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0488 - accuracy: 0.9856
    Epoch 5/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0372 - accuracy: 0.9890
    




    <keras.callbacks.History at 0x1ff057de190>



The final accuracy was 98.9%

**Using the model to make predictions**

Now that the model has been trained via the .fit method, we can now use the model and it's associated parameters/weights to predict new samples. 


```python
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]
```




    array([3.49187546e-09, 1.28081851e-10, 7.26641076e-07, 9.56786389e-05,
           1.91619528e-12, 1.10672689e-08, 2.23948379e-14, 9.99898076e-01,
           1.29563835e-08, 5.47320997e-06], dtype=float32)



`predictions` is now an array of arrays, each element of the highest level array has the probability for each number held within a 10-element array. To find the predicted number, we find the index of the maximum value via the `.argmax()` method. 


```python
predictions[0].argmax()
```




    7



So now we can compare the two cases between the predicted value (7) vs. the correct label for that datapoint. 


```python
predictions[0][7]
```




    0.9998981




```python
test_labels[0]
```




    7



The two are the same!

We can now plot the first ten test items along with their predictions. 


```python
import matplotlib.pyplot as plt
num = 10
images = test_images_2[:num]
labels = test_labels_2[:num]

predictions = model.predict(images.reshape((num, 28 * 28)))

num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}\nPredict: {}'.format(labels[i], predictions[i].argmax()))
plt.tight_layout()
plt.show()
```


    
<img src="/assets/images/DLwPCh2/output_30_0.png">
    


## Find examples where it predicted wrong and visualize them


```python
# Make predictions for every test image
all_predictions = model.predict(test_images.reshape((10000, 28 * 28)))

#grab the images where the predictions don't match the label
predicted_labels = np.argmax(all_predictions, axis=1)
failed_items = np.not_equal(predicted_labels,test_labels)
test_fails = test_images[failed_items]

#same indices used for predictions and labels 
prediction_fails = predicted_labels[failed_items]
label_fails = test_labels[failed_items]

num = 50

num_row = 5
num_col = 10# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(test_fails[i].reshape(28, 28), cmap='gray')
    ax.set_title('Label: {}\nPredict: {}'.format(label_fails[i], prediction_fails[i]))
plt.tight_layout()
plt.show()
```


    
<img src="/assets/images/DLwPCh2/output_32_0.png">
    




Getting into these failures can help to diagnose and direct next steps for improving the algorithm. 

Some of these example look very difficult!


**Evaluating the model on new data**

Get the accuracy rates for the full 10k test images.


```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
```

    313/313 [==============================] - 0s 818us/step - loss: 0.0669 - accuracy: 0.9797
    test_acc: 0.9797000288963318
    

If we compare this accuracy on the test set vs. the training set, we got 97.92% on this whereas we scored 98.9% on the training data. This is a textbook example of the system overfitting onto the training data. More on overfitting will be in the [next part](https://robertpiazza.com/deep_learning/2022/02/18/Deep-Learning-Part-3.html). 

Next we'll learn about tensors, tensor operations, and gradient descent.

## Data representations for neural networks

An array of (usually) numbers (such as the numpy arrays above) is called a **tensor**. 

The dimensionality of the tensor gives its rank and lower rank tensors have common names:
- Rank-0 tensor (single number) is a constant
- Rank-1 tensor (list or column) is a vector
- Rank-2 tensor (table) is a matrix

In the context of tensors, a dimension is typically called an  **axis**. 

### Scalars (rank-0 tensors)

single number as an array


```python
import numpy as np
x = np.array(12)
x
```




    array(12)




```python
#number of dimensions of the array
x.ndim
```




    0



### Vectors (rank-1 tensors)


```python
x = np.array([12, 3, 6, 14, 7])
x
```




    array([12,  3,  6, 14,  7])




```python
x.ndim
```




    1



### Matrices (rank-2 tensors)


```python
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
x.ndim
```




    2



#### Rows and Columns
- 5, 78, 2, 34, 0 is the first row,
- 5, 6, 7 are the first column

### Rank-3 and higher-rank tensors

- This Rank-3 tensor can be thought of as a rectangular prism of numbers.

- Most ML is done on ranks 0-4 tensors but video may be up to rank 5. 


```python
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
x.ndim
```




    3



### Key attributes

3 attributes define a tensor:

1. Number of axes
2. Shape
3. Data type (dtype)




```python
#load the dataset
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```


```python
#number of axes
train_images.ndim
```




    3




```python
#tensors shape
train_images.shape
```




    (60000, 28, 28)




```python
#data types held inside
train_images.dtype
```




    dtype('uint8')



**Displaying the fourth digit**

Using Matplotlib to show the 4th of the 60000 training examples


```python
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```


    
<img src="/assets/images/DLwPCh2/output_55_0.png">
    



```python
#and its label is:
train_labels[4]
```




    9



### Manipulating tensors in NumPy

different views or sub-parts of a tensor/array are called **slices**


```python
my_slice = train_images[10:100] 
#note this doesn't include the element at 100
my_slice.shape
```




    (90, 28, 28)




```python
#this is the same operation as before but more detailed
#the colon says include everything in the dimension the colon is located

my_slice = train_images[10:100, :, :]
my_slice.shape
```




    (90, 28, 28)




```python
#you could also just say everything from 0 to 27 for those last two dimensions

my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape
```




    (90, 28, 28)




```python
#matplotlib counts from the top-left to bottom-right horizantally
#The following would select the bottom right of the images
my_slice = train_images[:, 14:, 14:]
```


```python
#this would crop the middle square of the images
#the negative indices indicate a relative distance from the end of the tensor
my_slice = train_images[:, 7:-7, 7:-7]
```

### The notion of data batches

By convention, the first axis/dimension of the tensor keeps each individual training example or **sample**

The first axis can also be called the **batch dimension**

Deep learning models won't train on the entirety of the dataset at once. They do it on small batches of the training set at a time. 

Here we train on the first 128 images


```python
batch_size = 128
batch = train_images[:batch_size]
```

Here we train on the next 128 images


```python
batch = train_images[batch_size:(batch_size*2)]
```

This is generalizable for the nth batch:


```python
n = 3
batch = train_images[batch_size * n:batch_size * (n + 1)]
```

### Real-world examples of data tensors

- Vectors `(samples, features)`
- Timeseries `(samples, timesteps, features)`
- Images `(samples, height, width, channels)`
- Video `(samples,frames,height,width,channels)`

### Vector data
- we technically reshaped the MNIST dataset from a Rank-3 dataset to Rank-2 when we changed the 28x28 to 784 size. 
- The cars dataset with each car being a sample and information about make, model, year, mpg, engine size, etc. Same for the iris dataset

### Timeseries data or sequence data

- time axis is always the second axis by convention

- an example would be a dataset of stock prices throughout the day or year


### Image data

- images have samples, height, width, and three color channels

- 200 color images of size 100x100 could be stored in a tensor of shape `(200, 100, 100, 3)`

- By convention, the color channel either comes first after the sample dimension or last - most datasets carry the color channel last. 



### Video data

- A set of 10 movie clips, each 30 seconds long, sample 10 times per second, and sized 128x256, could be represented with a tensor of shape `(10, 300, 128, 256, 3)`. 

## The gears of neural networks: tensor operations

- All input transformations can be reduced to a few different tensor operations/functions

The first example neural net we created was the dense layer:

`keras.layers.Dense(512, activation="relu")`

The dense layer is performed with the following function:

`output = relu(dot(input, W) + b)`

There's three operations going on here:

1. Dot product of input tensor and tensor "W"

Quick dot product reference:
$\begin{bmatrix} 
	1 & 2 \\
	3 & 4 \\
	10 & 0 \\
	\end{bmatrix} * [3, -1] = [[1,2]*[3,-1]+[3,4]*[3,-1]+[10,0]*[3,-1]$
    $=[1, 5, 30]$


2. Adding that matrix with vector b (the bias)

$[1, 5, 30]+[3, 12, -32]=[4, 17, -2]$

3. Performing a relu operation - max(x,0) and stands for "rectified linear unit"

relu($[4, 17, -2]$)=$[4, 17, 0]$

### Element-wise operations

The following functions use nested for loops to go through each element of their tensors and apply relu and addition of two vectors. 

This shows how the math is performed for these operations but our numerical libraries can perform them much faster via vectorized and parallel operations than these for loops ever could.

The difference in time is shown for the final two examples, the first has been vectorized using the numpy library, the second uses the for loops. These times add up significantly for large training sets. 

My machine ran the numbers in 0.0040 seconds versus the for loop of 1.77 seconds. A speedup factor of 442.5! 


```python
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
```


```python
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x
```


```python
import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)
print("Took: {0:.4f} s".format(time.time() - t0))
```

    Took: 0.0040 s
    


```python
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))
```

    Took: 1.74 s
    

### Broadcasting

When vectors don't share the same shape, smaller shape will be broadcast across the larger one and reused. 


```python
import numpy as np
X = np.random.random((32, 10))
y = np.random.random((10,))
```


```python
y = np.expand_dims(y, axis=0) #y becomes a (1,10) tensor
```


```python
Y = np.concatenate([y] * 32, axis=0) #new Y for broadcasting
```


```python
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

#This function can now operate on the variables X and our new Y
```


```python
import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)

#This broadcasts y multiple times to compare to x across it's dimensions
```

### Tensor product (dot product)

- for two tensors to be multiplied, they must share the same size for their inner dimensions i.e. a (3,2)•(2,1) will result in a (3,1) vector but you can't flip them and perform (2,1)•(3,2) because the 1 and 3 don't match. 



```python
#Implementing a dot product in numpy

x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)
```


```python
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
```


```python
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
```


```python
def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z
```


```python
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z
```

<img src="/assets/images/DLwPCh2/Dot-Product.png">

Following rules and shapes apply for dot products of tensors of higher dimensions: 

(a, b, c, d) • (d,) → (a, b, c)

(a, b, c, d) • (d, e) → (a, b, c, e)

### Tensor reshaping


```python
train_images = train_images.reshape((60000, 28 * 28))
```


```python
x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])
x.shape
```




    (3, 2)




```python
x = x.reshape((6, 1))
x
```




    array([[0.],
           [1.],
           [2.],
           [3.],
           [4.],
           [5.]])




```python
x = np.zeros((300, 20))
x = np.transpose(x) #exchange rows for columns
x.shape
```




    (20, 300)



### Geometric interpretation of tensor operations

### A geometric interpretation of deep learning

## The engine of neural networks: gradient-based optimization

### What's a derivative?

### Derivative of a tensor operation: the gradient

### Stochastic gradient descent

### Chaining derivatives: The Backpropagation algorithm

#### The chain rule

#### Automatic differentiation with computation graphs

#### The gradient tape in TensorFlow


```python
import tensorflow as tf
x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
```


```python
x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
```


```python
W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])
```

## Looking back at our first example


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```


```python
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```


```python
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```


```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    469/469 [==============================] - 2s 3ms/step - loss: 0.2573 - accuracy: 0.9257
    Epoch 2/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.1044 - accuracy: 0.9696
    Epoch 3/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0688 - accuracy: 0.9789
    Epoch 4/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0507 - accuracy: 0.9848
    Epoch 5/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0381 - accuracy: 0.9882
    




    <keras.callbacks.History at 0x1ff1529b880>



### Reimplementing our first example from scratch in TensorFlow

#### A simple Dense class


```python
import tensorflow as tf

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]
```

#### A simple Sequential class


```python
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)
        return x

    @property
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights
```


```python
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4
```

#### A batch generator


```python
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

### Running one training step


```python
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
```


```python
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)
```


```python
from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```

### The full training loop


```python
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")
```


```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)
```

    Epoch 0
    loss at batch 0: 9.22
    loss at batch 100: 2.22
    loss at batch 200: 2.21
    loss at batch 300: 2.08
    loss at batch 400: 2.25
    Epoch 1
    loss at batch 0: 1.93
    loss at batch 100: 1.87
    loss at batch 200: 1.83
    loss at batch 300: 1.71
    loss at batch 400: 1.86
    Epoch 2
    loss at batch 0: 1.61
    loss at batch 100: 1.57
    loss at batch 200: 1.51
    loss at batch 300: 1.43
    loss at batch 400: 1.53
    Epoch 3
    loss at batch 0: 1.35
    loss at batch 100: 1.34
    loss at batch 200: 1.24
    loss at batch 300: 1.22
    loss at batch 400: 1.30
    Epoch 4
    loss at batch 0: 1.15
    loss at batch 100: 1.15
    loss at batch 200: 1.04
    loss at batch 300: 1.05
    loss at batch 400: 1.13
    Epoch 5
    loss at batch 0: 1.00
    loss at batch 100: 1.01
    loss at batch 200: 0.90
    loss at batch 300: 0.93
    loss at batch 400: 1.01
    Epoch 6
    loss at batch 0: 0.89
    loss at batch 100: 0.91
    loss at batch 200: 0.80
    loss at batch 300: 0.84
    loss at batch 400: 0.92
    Epoch 7
    loss at batch 0: 0.81
    loss at batch 100: 0.82
    loss at batch 200: 0.72
    loss at batch 300: 0.77
    loss at batch 400: 0.85
    Epoch 8
    loss at batch 0: 0.74
    loss at batch 100: 0.75
    loss at batch 200: 0.66
    loss at batch 300: 0.72
    loss at batch 400: 0.80
    Epoch 9
    loss at batch 0: 0.69
    loss at batch 100: 0.70
    loss at batch 200: 0.61
    loss at batch 300: 0.67
    loss at batch 400: 0.75
    

### Evaluating the model


```python
predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
```

    accuracy: 0.82
    

## Summary
