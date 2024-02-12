# Employee-Attrition
This project delves into the root causes of employee attrition within the company. By analyzing data, and identifying trends, we aim to understand why valuable team members leave. Equipped with these insights, we can develop targeted strategies to improve employee retention, boost morale, and ultimately, enhance workplace culture .

## Libraries imported <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/2b53793b-1392-4719-a527-a80c3fda91df)

## Getting one with data <br>
Visualizing the dataset
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/c0e11036-7ed4-489a-83fb-4a18bacc718d)
<br>
Information about all the columns, non null values and the datatype present in each column
<br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/3d3d31dc-b259-4b6c-922a-8ee029502b14)
<br>
Visualizing attrition based on the gender 
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/105cc91b-ff1f-4d89-87fa-6ea49530e6e9)

<br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/6ffbcc6e-da24-4962-b8a9-46c9fe9f5668)

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/c1f7cf7d-6081-4cda-bad7-6739ea45b38b)

## Removing Unimportant Features
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/81a0ad23-09de-472c-ab0b-4a27a6b1866f)

## Selecting X, y
<br>
X = all the features important for considering the classification, in this case "Whether an employee can be retened or not?
y = include the attrition features consisting of values = 'yes,no"

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/70570b0f-6c56-481d-b8bb-e2c6bf22a76e)

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/38b98b5e-a538-4fbc-bd50-6a7c83cd2d4f)

## Feature encoding
<br>
Since machine learning models can understand only the numerical data, therefore it is necessary to encode X and y.

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/97b9a56e-dda0-4f0c-ab98-560a75587a87)

Columns after encoding 
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/2ae68c16-6cf5-4b92-8598-43302145c41c)

## Balancing the Imbalanced data <br>
Imbalanced data
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/2874a401-b7ef-4c7e-843a-94392d48e7d8)

Balancing using SMOTE <br> 
SMOTE - SMOTE, or Synthetic Minority Oversampling Technique, is a widely used technique in machine learning to address the problem of class imbalance. In datasets where one class (usually the minority class) has significantly fewer samples than the others, traditional machine learning algorithms can become biased towards the majority class and perform poorly on the minority class.
<br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/1a3fdb28-5ce5-44d4-ab4c-bbe1636a1f09)

## Feature Standardization
<br>
Feature standardization, also known as z-score normalization, is a technique commonly used in machine learning to prepare data for modeling. It involves transforming each feature in your dataset to have a mean of 0 and a standard deviation of 1. One of the main reason to standardize feature is to - Improves algorithm performance.
<br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/0b76a25e-1ba4-476d-a008-68227b3443f8)

## Spliting the data <br>
Data is to be splitted into test and train using sklearn.preprocessing.train_test_split
<br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/6795fd1b-0fde-4917-8054-3dcff17e73d1) <br>
<br>
Shape of splitted data
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/a6fd6c0e-e34f-4d82-b36d-dbc1235c7c48)

## Fitting the data into models <br>
Various Models have been used to understand the relationship of various features with the Attrition values. Classification models like
1. Logistic Regression
2. Random forest Classifier
3. KNN
4. Ensembling model - Gradient Boosting

## Efficiency of the model <br>
Various metrics like :- 
1. Confusion Matrix <br>
It is a square table with rows and columns representing the actual and predicted classes, respectively. Each cell displays the number of data points belonging to a specific combination of actual and predicted classes.

Key Values:

True Positive (TP): Correctly predicted positive cases. <br>
True Negative (TN): Correctly predicted negative cases. <br>
False Positive (FP): Incorrectly predicted positive cases (Type I error). <br>
False Negative (FN): Incorrectly predicted negative cases (Type II error). <br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/903bad11-b03a-4dd8-9f2a-a2a8dd352607) <br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/a40c9ff1-9b6f-44f1-b5b1-426f1452ecaa)

2. Classification Report <br>
It provides a detailed and informative summary of how well the classification model performs on each class in the dataset. It acts as a valuable tool for evaluating and understanding the strengths and weaknesses of your model.
<br>
It includes: <br>
a. Precision: This metric tells you how many of the predicted positive cases were actually positive. It represents the proportion of true positives among all positive predictions. <br>
b. Recall: This metric indicates how many of the actual positive cases were correctly identified by the model. It represents the proportion of true positives among all actual positive cases. <br>
c. F1-score: This metric is a harmonic mean of precision and recall, combining both metrics into a single score. <br>
d. Support: This metric represents the total number of data points belonging to a specific class.

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/372c037d-12ef-4579-9901-cab07c44f008)

3. ROC Curve
The Receiver Operating Characteristic (ROC) curve is a widely used visual tool in machine learning for evaluating the performance of binary classification models. It provides a comprehensive insight into a model's ability to distinguish between positive and negative classes.
ROC curve plots the True Positive Rate (TPR) on the y-axis against the False Positive Rate (FPR) on the x-axis.
True Positive Rate (TPR): The proportion of actual positives correctly identified by the model (also known as recall).<br>
False Positive Rate (FPR): The proportion of actual negatives incorrectly identified as positives (also known as 1 - specificity).
<br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/0ead6317-aa42-4d29-9ca5-1664be150939)

## Hypertuning the models <br>
Hyperparameter tuning, also known as hyperparameter optimization, is an essential step in machine learning to get the most out of the model. It involves adjusting the settings that control the model's learning process to achieve the best possible performance on your specific task. <br>
Advantages:- <br>
1. Improved model performance: By finding the optimal hyperparameter values, you can significantly boost your model's accuracy, precision, recall, or other relevant metrics. This can be especially important for real-world applications where even small performance gains can have a substantial impact. <br>
2. Reduced overfitting/underfitting: Hypertuning helps to strike a balance between overfitting, where the model memorizes the training data too well but performs poorly on unseen data, and underfitting, where the model fails to capture the underlying patterns in the data. <br>
3. Enhanced generalizability: By finding settings that work well across different datasets or splits of the same data, you ensure that your model will perform well on new, unseen data, not just the training data it was initially trained on. <br>

Example of Hypertuning a model:- <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/32a4104f-71fb-4aa6-a19e-5d00c17153a9)


## Check for overfitting of the model <br>
Overfitting is a common concern in machine learning, where the model memorizes the training data too well and performs poorly on unseen data.
There are various ways to check for underfitting but in our model, we have focused on cross validation score.
<br>
### Cross validation Score <br>

In machine learning, cross-validation (CV) is a technique used to estimate the generalizability of a model on unseen data.
The cross-validation score summarizes the performance of a model across multiple training and testing splits of the data. It helps answer the question: "How well would this model perform on new data it hasn't seen before?" <br>
It reduces overfitting -  by training and testing on different subsets of the data, CV prevents the model from memorizing the specific characteristics of the training data, leading to better generalization. <br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/a0848af1-616b-44c0-b790-615417c0c53f)


## Extracting Important Features <br>
Feature importance analysis helps to comprehend how the model makes predictions. By identifying the features with the highest impact on the outcome, you gain insights into the relationships between features and the target variable.
This understanding improves interpretability and trust in the model, especially for complex algorithms like neural networks.
<br>
1. Logistic Regression <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/3fe6e0d9-56ca-4130-b77b-6ad213060369)
<br>
2. Decision Tree <br>

![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/239ce256-2008-4281-b2b9-874398dd5d90)

<br>

3. Gradient Bosting <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/63696985-53d2-4af9-bd89-6be15fd57f3f)


## Selecting the best model <br>
Out of 4 different model used for classification problem. The best among it was KNN. <br>
The confusion matrix of KNN <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/e1480638-a7da-49a5-af46-78b7a01fe0b9)
<br>

Confusion matrix of other models :- <br>
1. Logistic regression <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/939230c5-6edd-424d-8af6-c90db739d465)

2. Decision Tree Classifier <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/d5d71619-ea28-4ded-8b2d-b966d552e7b3)

3. Gradient Boosting <br>
![image](https://github.com/shashank-2010/Employee-Attrition/assets/153171192/6082db24-0156-4b0a-b31a-b6a9bfef0e2f)













