#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')


# In[2]:


df_emp = pd.read_csv(r'D:\Internship Project\Attrition data.csv')
df_emp.head(3)


# In[3]:


df_emp.info()


# In[4]:


df_emp.isnull().sum()


# In[5]:


df_emp.dropna(inplace=True)


# In[6]:


df_emp.duplicated().sum()


# In[7]:


plt.bar(df_emp.Attrition.unique(), df_emp.Attrition.value_counts())
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.title('Distribution of Attrition in df_emp')
plt.show()


# IT IS AN IMBALANCED SAMPLE

# In[8]:


df_emp.Attrition.value_counts()


# In[9]:


df_emp.Gender.value_counts()


# In[10]:


df_emp.groupby('Gender')[['Attrition']].value_counts()


# In[11]:


import matplotlib.pyplot as plt

# Data provided
data = {
    'Female': {'No': 1464, 'Yes': 265},
    'Male': {'No': 2141, 'Yes': 430}
}

# Extract genders and counts for clarity
genders = list(data.keys())
no_counts = [data[gender]['No'] for gender in genders]
yes_counts = [data[gender]['Yes'] for gender in genders]

# Create the bar plot
bar_width = 0.35 
index = range(len(genders))  # Create x-axis ticks

plt.figure(figsize=(8, 6))

# Plot "No" bars
plt.bar(index, no_counts, bar_width, label='No', color='skyblue')

# Plot "Yes" bars on top, shifted towards right
plt.bar([i + bar_width for i in index], yes_counts, bar_width, label='Yes', color='coral')

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of "No" and "Yes" Values by Gender')

# Add x-axis labels
plt.xticks([i + bar_width / 2 for i in index], genders)

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[12]:


df_emp.BusinessTravel.value_counts()


# In[13]:


df_emp.drop(columns=['EmployeeID','EmployeeCount'], inplace = True)


# In[14]:


X_features = list(df_emp.columns)
X_features.remove('Attrition')
X_features


# In[15]:


#encoding the data in X_features into numerical form using One Hot Encoding
encoded_df_emp = pd.get_dummies(df_emp[X_features], drop_first=True)
X = encoded_df_emp
X.head()


# In[16]:


X.columns


# In[17]:


X.Age.unique()


# In[18]:


X.MonthlyIncome.describe()


# In[19]:


px.box(X.MonthlyIncome)


# In[20]:


#encoding y
y = df_emp.Attrition.map(lambda x:1 if x == 'Yes' else 0)
y.value_counts()


# In[21]:


#Resampling X,y
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X,y = sm.fit_resample(X,y)


# In[22]:


X.info()


# In[23]:


y.value_counts()


# In[24]:


#standardizing X
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
X_scaled


# In[25]:


#data training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, train_size=0.75, random_state=42)


# In[26]:


X_train,y_train


# In[27]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # Using Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg_model = logreg.fit(X_train,y_train)


# In[29]:


y_pred = logreg_model.predict(X_test)
y_pred


# In[30]:


logreg_df = pd.DataFrame({'Actual_value':y_test,
                         'Pred_value':y_pred})
logreg_df.sample(5)


# In[31]:


from sklearn.metrics import confusion_matrix,classification_report

def draw_cm(actual,predicted):
    cm = confusion_matrix(actual,predicted)
    sns.heatmap(cm, annot=True, fmt='.2f',
               xticklabels=['NO','YES'],
               yticklabels=['NO','YES'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[32]:


draw_cm(logreg_df.Actual_value, logreg_df.Pred_value)


# In[33]:


print(classification_report(logreg_df.Actual_value,logreg_df.Pred_value))


# In[34]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)


# In[35]:


feature_rank_logreg = pd.DataFrame({'features':X.columns,
                                   'importance':logreg_model.coef_.reshape(39)})

feature_rank_logreg = feature_rank_logreg.sort_values('importance', ascending=False)

plt.figure(figsize=(8,8))
sns.barplot(y='features', x='importance', data=feature_rank_logreg);


# In[36]:


#check for overfitting
from sklearn.model_selection import cross_val_score

cv_score_logreg = cross_val_score(logreg_model, X_train,y_train, cv=10, scoring='roc_auc')
cv_score_logreg


# In[37]:


print('Mean Accuracy',round(np.mean(cv_score_logreg),3), 'Standard Deviation',round(np.std(cv_score_logreg),3))


# In[38]:


#draw Roc_curve
from sklearn import metrics

def draw_roc_curve(model,X_test,y_test):
    test_result_df = pd.DataFrame({'actual':y_test})
    test_result_df = test_result_df.reset_index()
    
    pred_prob_df= pd.DataFrame(model.predict_proba(X_test))
    
    test_result_df['classlabel1'] = pred_prob_df.iloc[:,1:2]
    
    fpr,tpr,thresholds = metrics.roc_curve(
                                            test_result_df.actual,
                                            test_result_df.classlabel1,
                                            drop_intermediate = False)
    
    auc_score = roc_auc_score(test_result_df.actual, test_result_df.classlabel1)
    
    plt.figure(figsize = (8,6))
    
    plt.plot(fpr,tpr,label = 'Roc curve(area=%0.2f)' % auc_score)
    
    plt.plot([0,1],[0,1],'--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Postive Rate or [1-TPR]')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
    
    return auc_score, fpr,tpr,thresholds


# In[39]:


draw_roc_curve(logreg_model,X_test,y_test)


# ## Using Decision Tree

# In[40]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion='gini',
                                max_depth=3,
                                min_samples_split=2)
dectree_model = dectree.fit(X_train,y_train)


# In[41]:


y_pred_tree = dectree_model.predict(X_test)
y_pred_tree


# In[42]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_tree)


# In[43]:


draw_roc_curve(dectree_model, X_test, y_test)


# In[44]:


#hypertuning the parameters
from sklearn.model_selection import GridSearchCV
tuned_params = [{'criterion':['gini','entropy'],
                'max_depth': range(2,5)}]


# In[45]:


clf_tree = DecisionTreeClassifier()
clf=GridSearchCV(clf_tree, tuned_params, cv=10, scoring='roc_auc')


# In[46]:


clf.fit(X_train,y_train)


# In[47]:


clf.best_params_


# In[48]:


clf.best_score_


# In[49]:


clf_best = DecisionTreeClassifier(criterion='gini',
                                 max_depth=4)
clf_best.fit(X_train,y_train)


# In[50]:


y_pred_best_tree = clf_best.predict(X_test)


# In[51]:


roc_auc_score(y_test,y_pred_best_tree)


# In[52]:


draw_roc_curve(clf_best,X_test,y_test)


# In[53]:


dectree_df = pd.DataFrame({'Actual_value':y_test,
                          "Pred_value":y_pred_best_tree})
dectree_df.head(4)


# In[54]:


draw_cm(dectree_df.Actual_value,dectree_df.Pred_value)


# In[55]:


print(classification_report(dectree_df.Actual_value,dectree_df.Pred_value))


# In[56]:


#importance of feature in the model
feature_rank_tree = pd.DataFrame({'features':X.columns,
                                 'importance':clf_best.feature_importances_})
feature_rank_tree = feature_rank_tree.sort_values('importance', ascending=False)

plt.figure(figsize=(8,8))
sns.barplot(y='features', x='importance', data= feature_rank_tree);


# In[57]:


#check for overfitting
from sklearn.model_selection import cross_val_score

cv_score_tree = cross_val_score(clf_best, X_train,y_train, cv=10, scoring='roc_auc')
cv_score_tree


# In[58]:


print('Mean Accuracy',round(np.mean(cv_score_tree),3), 'Standard Deviation',round(np.std(cv_score_tree),3))


# ## Using KNN classification

# In[59]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,
                          weights = 'uniform',
                          algorithm = 'auto',
                          metric = 'minkowski')
knn_model = knn.fit(X_train,y_train)


# In[60]:


y_pred_knn = knn_model.predict(X_test)
y_pred_knn


# In[61]:


roc_auc_score(y_test,y_pred_knn)


# In[62]:


#hypertuning
tuned_params_knn = [{'n_neighbors':range(4,8),
                    'weights':['uniform','distance'],
                    'algorithm':['auto','kd_tree','brute'],
                    }]
knn_1 = KNeighborsClassifier()
knn_hypt = GridSearchCV(knn_1, tuned_params_knn, cv=10, scoring='roc_auc')


# In[63]:


knn_hypt.fit(X_train,y_train)


# In[64]:


knn_hypt.best_params_


# In[65]:


knn_best = KNeighborsClassifier(n_neighbors=7,
                          weights = 'distance',
                          algorithm = 'kd_tree',
                          metric = 'minkowski')
knn_best_model = knn_best.fit(X_train,y_train)


# In[66]:


y_pred_knn_best = knn_best_model.predict(X_test)
y_pred_knn_best


# In[67]:


roc_auc_score(y_test,y_pred_knn_best)


# In[68]:


draw_roc_curve(knn_best_model, X_test, y_test)


# In[69]:


knn_df = pd.DataFrame({'Actual_value':y_test,
                      'Pred_value':y_pred_knn_best})
knn_df.head(3)


# In[70]:


draw_cm(knn_df.Actual_value,knn_df.Pred_value)


# In[71]:


print(classification_report(knn_df.Actual_value,knn_df.Pred_value))


# In[72]:


#check for overfitting
from sklearn.model_selection import cross_val_score

cv_score_knn = cross_val_score(knn_best_model, X_train,y_train, cv=10, scoring='roc_auc')


# In[73]:


cv_score_knn


# In[74]:


print('Mean Accuracy',round(np.mean(cv_score_knn),3), 'Standard Deviation',round(np.std(cv_score_knn),3))


# In[75]:


train_score = knn_best_model.score(X_train, y_train)
val_score = knn_best_model.score(X_test, y_test)
train_score, val_score


# ## Using Ensemble - Boosting

# In[76]:


from sklearn.ensemble import GradientBoostingClassifier
grad_en = GradientBoostingClassifier(loss = 'deviance',
                                    n_estimators=100,
                                    criterion='mse',
                                    max_depth=5,
                                    random_state=42,
                                    max_features='log2')
grad_en_model = grad_en.fit(X_train,y_train)


# In[77]:


y_pred_grad = grad_en_model.predict(X_test)
y_pred_grad


# In[78]:


roc_auc_score(y_test,y_pred_grad)


# In[79]:


draw_roc_curve(grad_en_model,X_test,y_test)


# In[80]:


grad_df = pd.DataFrame({'Actual_value':y_test,
           'Pred_value':y_pred_grad})
grad_df.head(3)


# In[81]:


draw_cm(grad_df.Actual_value,grad_df.Pred_value)


# In[82]:


print(classification_report(grad_df.Actual_value,grad_df.Pred_value))


# In[83]:


#importance of feature in the model
feature_rank_grad = pd.DataFrame({'features':X.columns,
                                 'importance':grad_en_model.feature_importances_})
feature_rank_grad = feature_rank_grad.sort_values('importance', ascending=False)

plt.figure(figsize=(8,8))
sns.barplot(y='features', x='importance', data= feature_rank_grad);


# In[84]:


#check for overfitting
from sklearn.model_selection import cross_val_score

cv_score_grad = cross_val_score(grad_en_model, X_train,y_train, cv=10, scoring='roc_auc')


# In[85]:


#cv_score across 10 cross validation cycle
cv_score_grad


# In[86]:


print('Mean Accuracy',round(np.mean(cv_score_grad),3), 'Standard Deviation',round(np.std(cv_score_grad),3))


# In[87]:


train_score_grad = grad_en_model.score(X_train, y_train)
val_score_grad = grad_en_model.score(X_test, y_test)
train_score_grad, val_score_grad


# In[ ]:




