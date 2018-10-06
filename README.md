# Text-Classification


## Abstract:
In this project, three methods were used to classify emails. The classification is done using kNN, Naive Bayes and Artificial Neural Network. First, image data is either normalized or scaled. Then it is fed to classifier. Note the only purpose of this study is to compare various methods of image classification.


## Methadology:

### 1) Data Set Selection:
Dataset used here is a subset of Enron Email Dataset provided by Enron Cooperation. The subset contains 33687 emails out of which 16545 are not spam/ham and 17142 are spam. This dataset can found in my [Google Drive](https://drive.google.com/open?id=18TVvrPHDs-8Ww7kPbZpJH-5Vuuol2H_X). This dataset's main purpose is to compare different approaches of text classification. The code can be run on any dataset. The only difference is that first line (subject line) of every email is removed. It can be changed by removing line 18: handle.__next__() of ReadPreprocessData.py.

### 2) Feature Selection:
Each word in email is considered a feature.

### 3) Data Pre-processing:
Following preprocessing techniques were used in order:
* Convert to String
* Convert to Lowercase
* Remove numbers and special characters
* Remove stop words
* Convert to sparse vector

### 4) Machine Learning Algorithm:
As we were using supervised learning approach to classify, we used kNN, Naive Bayes and Artificial Neural Network.

## Results:
We got good results from Naive Bayes and Artificial Neural Network the methods. kNN failed to produce any results.

## Report
Detailed report is given in [Project-Report](../master/Project-Report.docx).


## Conclusion:
To conclude, all methods work for image classification. However, CNN gives great results with minimum prediction time.


## Contact
You can get in touch with me on my LinkedIn Profile: [Farhan Shoukat](https://www.linkedin.com/in/farhan-shoukat-782542167/)


## License
[MIT](../master/LICENSE)
Copyright (c) 2018 Farhan Shoukat
