---
layout: post
title:  "Jupyter Environments"
date:   2022-02-15
categories: deep_learning
---
# How to Create a New Python Environment in Anaconda and Use it in Jupyter Notebooks

## Start Anaconda Prompt


```python
#(in base environment) 
conda install nb_conda_kernels

#make sure there's no spaces in the new environment's name
conda create --name environment-name 

conda activate environment-name
conda install ipykernel

#install whatever packages you need (versions optional)
conda install tensorflow==2.6

#when all packages are installed run
jupyter notebook
```

## In the Jupyter Notebook, navigate the menu to Kernel>Change kernel>conda env:environment-name
