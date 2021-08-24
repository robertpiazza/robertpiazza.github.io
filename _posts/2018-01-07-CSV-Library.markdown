---
layout: post
title: CSV Library
date: 2018-01-07
mathjax: true
categories:
tags:
---

### Standard CSV Library


```python
#Reading a CSV File
import csv
infile = open('1-1-1 sample.data','r') #the r means open for reading, use w when writing
reader = csv.reader(infile)
for line in reader:
    print(line)
infile.close()
```

    ['Field0', 'Field1', 'Field2', 'Field3']
    ['0', '1', '2', '3']
    ['text', '(also called strings)', 'can go', 'here too']
    ['pi', ' "e"', "Euler's constant", ' "e^(pi/2)"']
    ['3.14159', '2.71828', '0.577215', '0.207879576']
    ["You Can't", 'Have Your', 'Cake and', 'Eat it Too']


When Using quotes, you can use either singles (') or doubles (")
But if your text already includes one of them, use the other


```python
'He turned to me and said, "Hello there"'
```




    'He turned to me and said, "Hello there"'



#### Writing Data to a File

```python
import csv

#Open the file for writing ('w')
outfile = open('1-1-1 newfile.csv', 'w')

#Construct a writer
out = csv.writer(outfile, lineterminator='\n')

out.writerow(['this','is','your','header'])
for i in range(10):
    #Write the rows using the writer
    out.writerow([i,i+1,i+2,i+3])

#Close the file
outfile.close()

#Note: The string used to terminate lines produced by the writer defaults
#to \r\n. This may cause issues for non-Windows users if you do not know this
#is the default.

```

```python
## Creates a file with following contents:
this,is,your,header
0,1,2,3
1,2,3,4
2,3,4,5
3,4,5,6
4,5,6,7
5,6,7,8
6,7,8,9
7,8,9,10
8,9,10,11
9,10,11,12

```

Alternatively, most libraries contain a `.to_csv()` method you can use for exporting items to a csv or other tabular file

Revised example based on work first performed at ExploringDataScience.com
