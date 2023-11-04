# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Register Number: 212222230036
Name: Elamaran S E

import pandas as pd
df=pd.read_csv("/content/Churn_Modelling.csv")
df.head()
df.isnull().sum()
df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
```
## OUTPUT:
![279596621-02588e0a-67da-4ef9-9d92-50730ae40863](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/09861e9d-ef62-417d-8f00-32b32784d23e)

![279596722-42ceaa0a-2dec-46ad-9eb9-f0bde1c1cdbf](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/b4e5487d-e8f0-4726-8198-b96f84185f63)

![279596779-e371ae0b-994a-4725-90b3-5d0e788c97ca](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/9b3274e2-c8cb-428c-81b5-3f8ceb4ad587)

![279596867-3a5671bf-646e-4988-885e-6b0f68ef006b](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/fffe48fd-e722-4e7e-ab0c-a65f6768966e)

![279596968-0250b921-01a5-4094-9cbc-bbf496ebe6eb](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/9479ca6b-4dc2-4df9-9d89-9ea90632e9e8)

![279597058-1c2fc5a4-58ae-40e4-923d-4c686e6815a6](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/b39213a9-a25d-4986-94f3-b42939bc6d57)

![279597106-b14b84c3-5541-470a-801a-8848370e282c](https://github.com/elamarannn/Ex.No.1---Data-Preprocessing/assets/113497531/f73d33d9-613e-4b46-b23e-31661ef1c166)



## RESULT
Hence,the data preprocessing is done using the above code and data has been splitted into trainning and testing data for getting a better model.


