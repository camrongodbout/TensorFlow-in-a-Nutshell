TensorFlow in a Nutshell Part Two: Hybrid Learning
==================================================

Check out the article explaining all the in's and out's at [camron.xyz](http://www.camron.xyz)

To run this code simply type

    $ python wide.py

There are two dependencies for this:

 - Pandas
 - TensorFlow

The csv files were processed with preprocess.py

preprocess.py performs 3 actions

 1. Impute missing ages with the median age as an integer.
 2. Fill in the missing embarked with an S
 3. Fill missing Cabin with "None" since the TensorFlow hash bucket expects a string type and not a Nan.