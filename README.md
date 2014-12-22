Kaggle-DigitalRecognition
=========================

This is some code I wrote for Kaggle Competetion - [Digit Recognition](https://www.kaggle.com/c/digit-recognizer/).

Originally code were modified from Andrew Ng's course *Machine Learning*.

##	New Features:

1.	Customize hidden layer, user could add as many hidden layers as they want.
2.	Sharpen function, which would polarize the pixel value to be either 0 or 255, the default is 194.
3.	Principle component Analysis, so user should change the `K` value in `gg.m` file as their own.
4.	Automatically select lambda.
5.	Run `gg.m` first to gain see general results (both accuracy and F1 measurement).
6.	Then run `gt.m` or not as you wish, which would study the Theta values automatically.

##	By The Way

1.	The code was **only** tested under **Octave 3.8.1**
2.	All codes in this repository are under MIT License.

##	Demo

```
Last login: Mon Dec 22 22:06:40 on ttys000
Hasse-iMac:~ hasset$ tar xfj ~/....tar.bz2 -C /Volumes/RamDisk/ *.csv
Hasse-iMac:~ hasset$ octave
GNU Octave, version 3.8.1
Copyright (C) 2014 John W. Eaton and others.
This is free software; see the source code for copying conditions.
There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  For details, type 'warranty'.

Octave was configured for "x86_64-apple-darwin13.4.0".

Additional information about Octave is available at http://www.octave.org.

Please contribute if you find this software useful.
For more information, visit http://www.octave.org/get-involved.html

Read http://www.octave.org/bugs.html to learn how to submit bug reports.
For information about changes from previous versions, type 'news'.

octave:1> cd '~/Documents/MOOC/Kaggle/Digit Recognizer/SimpleNN'








Loading Data ...
Doing principle components analysis ...
Randomly spliting as training set, cv set and test set ...
Initialization competed, go or press Ctrl+C into manual mode ...
Selecting lambda ...
layer_size =

   100   500    10

n =  10
Iteration   100 | Cost: 1.260318e-01
Iteration   100 | Cost: 1.921399e-01
Iteration   100 | Cost: 1.169440e-01
Iteration   100 | Cost: 1.021654e-01
Iteration   100 | Cost: 2.012516e-01
Iteration   100 | Cost: 1.328409e-01
Iteration   100 | Cost: 1.306208e-01
Iteration   100 | Cost: 1.273289e-01
Iteration   100 | Cost: 2.008823e-01
Iteration   100 | Cost: 3.224707e-01
lambda		Train Error	Validation Error
 0.000000	0.126032	0.229367
 0.001000	0.192129	0.273496
 0.003000	0.116895	0.223911
 0.010000	0.101967	0.214513
 0.030000	0.200924	0.280045
 0.100000	0.131406	0.229149
 0.300000	0.126113	0.231509
 1.000000	0.107520	0.218947
 3.000000	0.165252	0.252027
 10.000000	0.237390	0.305760
lambda =  0.010000
Lambda selection completed, continue or press Ctrl+C into manual mode ...
Training ...
Iteration   500 | Cost: 7.872571e-04

The Accuracy for this Set is: 100.000000

The F1 for this Set is: 1.000000

The Accuracy for this Set is: 97.833333

The F1 for this Set is: 0.978459

The Accuracy for this Set is: 98.059524

The F1 for this Set is: 0.980774
Training completed, go onto manual mode or press Ctrl+C to quit.
Writing predicted results ...
Task completed.
octave:3> gt
Training ...
Iteration  1000 | Cost: 7.032029e-04

The Accuracy for this Set is: 100.000000

The F1 for this Set is: 1.000000

The Accuracy for this Set is: 98.119048

The F1 for this Set is: 0.981279

The Accuracy for this Set is: 98.119048

The F1 for this Set is: 0.981324
Training completed.
Writing predicted results ...
Task completed.
```