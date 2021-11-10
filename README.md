# Online Job Recruitment Scam Detection Using ML and NLP

Fake online job posting targeting people with the motive of stealing either money or personal information. The recruitment scams have not only led to peoples privacy issue but also has led to substantial financial loses. The motive of this project is to build a machine learning model that can predict a potential recruitment scam using NLP.

## Motivation

It was found that only a few researches were carried out in the field of recruitment scams. The only motive behind the previous studies was to increase accuracy. After trying to understand the business problem, it was found that in real-world scenarios machine learning should not only have high accuracy but should also have low latency. In this research, the main motive is to create a model with high accuracy with low latency. 

## Functionalities
    1. Exploratory data analysis
    2. Data sampling
    3. Data pre-processing
    4. Text Processing
    5. Feature Selection
    6. Encoding categorical variables
    7. Classification using ML algorithms
    8. Hyper-parameter tuning
    9. K-Fold cross validation
    10. Evaluation

## Architecture Diagram

![Picture1](https://user-images.githubusercontent.com/78141360/141083188-c8158bd3-488b-4ffe-b7d1-d0002e76f39f.png)
## Methodology
CRISP-DM is the methodology selected for the project. CRISP-DM stands for cross industry process for data mining. CRISP-DM methodology provides a fixed structured approach for planning a data mining project.

    1. Business understanding
    2. Data understanding
    3. Data preprocessing
    4. Modelling
    5. Evaluation
[https://drive.google.com/file/d/1JkFG77Z3CDEq9YmUuF-i1uV5rVvEm_p4/view?usp=sharing]

## Implementation

### 1. Data Selection
The dataset used in this project is downloaded from URL: http://emscad.samos.aegean.gr/.  
This dataset consists of 17866 rows and 18 columns.  
The dependent variable has two primary classes, namely, fraudulent or genuine.  
It is found that the data is highly imbalanced. The majority class has 1700 rows compared to only 866 rows of the minority class.  
Out of 18 variables, 2 contains text data like the job description and company profile. All other variables are either nominal or categorical. 

### 2. Exploratory Data Analysis
Exploratory data analysis refers to the critical process of conducting an initial investigation on any given data. It is done to spot anomalies, to discover underlying patterns, to test the hypothesis, and to check assumptions with the help of statistical and graphical representation.

### 3. Data Sampling
Class imbalance is considered to be a massive problem in classification problems. To handle the class imbalance problem, two methods can be used, upsampling and downsampling.  
In this case, the minority class has a decent amount of rows. Hence it was optimal to use downsampling for this project.  
Random downsampling is an approach in which the majority class is brought down to the size of the minority class. This method is used to solve the problem of class imbalance. To perform random down-sampling, no external libraries or pre-defined function were used.  
[https://drive.google.com/file/d/1flmggL-vymeLnhKakdH_YHtG49OQsSa8/view?usp=sharing]
[https://drive.google.com/file/d/1EV-b9Ohz848POGKPhDbQbFKfaqZP-FSs/view?usp=sharing]

### 4. Data Pre-processing
Data pre-processing is one of the most critical steps in machine learning. As, if the data is not pre-processed, it can add noise to the data which in return affect the model. All real-world data consists of missing values, un-wanted columns, inconsistent data, etc. In this project, the following data pre-processing steps were included. 
#### 4.1 Removing unwanted columns
Basic idea behind this step is to remove unwanted columns which can add noise to thedata. From the EDA (exploratory data analysis) section, it was found that all the categorical and nominal columns in the data had less than 0.5 correlation towards the dependent variable. This means that these columns don’t contribute much to the dependent variable and if kept, can increase unwanted noise to the model.  
Also ,columns with more than 75% null values were removed.
#### 4.2 Cleaning text data
**Removing non-English terms**: All terms other than English words are removed in this step. This step is essential because other terms in text data like special character and numbers can add noise to the data, which can adversely affect the performance of the machine learning model. A regular expression is used in this step to remove all non English terms.  
**Removing stop words**: Removing stop words is an essential step because stop words add dimensionality to the model; this additional dimensionality affects the performance of the model. Stopword package in the NLTK library is used for removing stop words in this project. All the text in the corpus is compared to the list of stop words, and if any word matches with the stop words list, it is then removed.  
**Lemmatizing words**: lemmatizing word is an essential step as this eliminated the problem of data duplications. Words with similar meaning such as work, working, and worked has the same meaning, but this will be considered as three different words while creating a bag of words model. WordNetLemmatizer package of NLTK library was used to tackle this problem. This package brings back any given words to its original form.  
**Normalizing cases**: normalizing the text is an essential step as this reduces the problem of dimensionality in the model. If the text is not normalized, it will lead to the problem of data duplication. For normalizing the text Lower () function in python is used. This function converts all the words into lower cases, which solves the problem.

### 5. Converting text data into numerical columns
**Bag of words** is a way of extracting features from any given text for working with machine learning algorithms. A bag of words model represents the text data according to the occurrence of words within a corpus.  
In simple terms, one hot encoding is done on the text data depending on the occurrence of the words.  
In terms of this project bag of words is implemented using CountVectorizer function of sklearn library. One of the specialties of CountVectorizer is that it can work as n-grams. In this project, we used uni-gram which is passed to the machine learning algorithms.  

### 6. Encoding Categorical variables
Machine learning models only understand number. So, it is vital to encode the data
in such a way that machine learning models can understand. There are many ways to
encode variables. The type of encoding used in this project is label encoding.  
**Label encoding**:
Label encoder labels a value between 0 to n-1 when n is the number of categories present. If a class repeats, it encodes the same value assigned before. In this project, the dependent variable is encoded. Label encoder is selected because there are only categories present and need not be one-hot encoding

### 7. Feature Selection
Feature selection is one of the most important steps in the field of text classification. As text data mostly have high dimensionality problem. To reduce the curse of high dimensionality, feature selection techniques are used. The basic idea behind feature selection is keeping only important features and removing less contributing features.  
**Chi-square Feature Selection**: Chi-Square test in statistics is used to check the independence of two events. In simple words, it tests whether the occurrence of a specific feature is independent to the class or not. The primary reason to choose chi-square feature selection was that it works well with categorical data. 

### 8. Classification using machine learning models
1. Random forest classifier without hyperparameter tuning
2. Random forest classifier with hyperparameter tuning
3. Support vector classifier
4. Naïve Bayes classifier 
5. XGBoost classifier
6. LightBGM classifier without hyperparameter tuning
7. LightBGM classifier with hyperparameter tuning

### 9. Evaluation
Evaluation is one of the most important part of any data mining project. This tells
how good a model is performing in terms of different metrics. The following evaluation methods are used in this project to evaluate the performance of the model.  
#### 9.1. Accuracy
Accuracy is the most commonly used metrics to evaluate the performance of any machine learning models. Accuracy is calculated using the following formula
#### 9.2. F1-Score
F1-Score is the harmonic mean of recall and precision. F1-score tell you how exact the classier is (how many cases it classfies correctly). The range of F1-score is from 0-1. F1-score is calculated by using the following formula
#### 9.3. Precision
Precision is the percentage of positive instance out of total predicted positive instance. In simple terms how much a given model is right when it says it is right. The precision of a model is calculated by using the following formula.
#### 9.4. Recall
Recall is the percentage of positive instance out of the total actual positive instance. In simple words, it is the true positive rate. Recall of a model is calculated by using the following formula.

### 10. Result
Gaussian Nave Bayes was found to be the best performing algorithm with high accuracy of 942363112 percentage. It also had one of the least execution time

![Unigram][https://drive.google.com/file/d/1qyNnreusl_ffIWrokK0NR7MOp5vNxnPG/view?usp=sharing]  
![Bigram][https://drive.google.com/file/d/17sbKBEwDV8GxmWYVkjmYPq4Jvhx80Bvr/view?usp=sharing]  
![Trigram][https://drive.google.com/file/d/1uW_d9gsXnyULBZZpyQ4Kdf2vuU9eOL4n/view?usp=sharing]  
![TF-IDF][https://drive.google.com/file/d/1pV9N32sQLDISkhVjQm7nO0rRWNXBqMGR/view?usp=sharing]  

### 11. Conclusion
The main motive behind this research was to develop a model which has not only high accuracy but also low latency. To attain these results, several steps were carried out. Firstly, all the categorical variables with low correlations were removed to reduce the space complexity of the model. After this step, the bag of words model was created, which had high dimensionality problem with sparse matrix. To handle this problem, ChiSquare feature selection technique was used. This step also increased the accuracy of the machine learning algorithm by removing all the noisy data. Gaussian Nave Bayes was found to be the best performing algorithm.
