def Traning_2(File):
    # Ignore  the warnings
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    # data visualisation and manipulation
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns
    import missingno as msno

    #configure
    # sets matplotlib to inline and displays graphs below the corressponding cell.
    # % matplotlib inline  
    style.use('fivethirtyeight')
    sns.set(style='whitegrid',color_codes=True)

    #import the necessary modelling algos.
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB

    #model selection
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
    from sklearn.model_selection import GridSearchCV

    from imblearn.over_sampling import SMOTE

    #preprocess.
    from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder

    # ann and dl libraraies
    from keras import backend as K
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
    from keras.utils import to_categorical
    from keras import utils as np_utils

    import tensorflow as tf
    import random as rn

    """## 1.2 ) Reading the data from a CSV file"""

    df=pd.read_csv(File)

    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    fig,ax = plt.subplots(5,2, figsize=(9,9))                
    sns.distplot(df['TotalWorkingYears'], ax = ax[0,0]) 
    sns.distplot(df['MonthlyIncome'], ax = ax[0,1]) 
    sns.distplot(df['YearsAtCompany'], ax = ax[1,0]) 
    sns.distplot(df['DistanceFromHome'], ax = ax[1,1]) 
    sns.distplot(df['YearsInCurrentRole'], ax = ax[2,0]) 
    sns.distplot(df['YearsWithCurrManager'], ax = ax[2,1]) 
    sns.distplot(df['YearsSinceLastPromotion'], ax = ax[3,0]) 
    sns.distplot(df['PercentSalaryHike'], ax = ax[3,1]) 
    sns.distplot(df['YearsSinceLastPromotion'], ax = ax[4,0]) 
    sns.distplot(df['TrainingTimesLastYear'], ax = ax[4,1]) 
    plt.tight_layout()
    # plt.show()

    """the various categorical features to use a count plot to show the relative count of observations of different categories."""

    cat_df=df.select_dtypes(include='object')

    cat_df.columns

    #corelation matrix.
    cor_mat= df.corr()
    mask = np.array(cor_mat)
    mask[np.tril_indices_from(mask)] = False
    fig=plt.gcf()
    fig.set_size_inches(30,12)


    df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
            ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)

    from sklearn import preprocessing
    def transform(feature):
        le=preprocessing.LabelEncoder()
        df[feature]=le.fit_transform(df[feature])
        print(le.classes_)

    cat_df=df.select_dtypes(include='object')
    cat_df.columns

    for col in cat_df.columns:
        transform(col)

    df.head() 
    scaler=preprocessing.StandardScaler()
    scaled_df=scaler.fit_transform(df.drop('Attrition',axis=1))
    X=scaled_df
    Y=df['Attrition'].to_numpy()

    from sklearn.model_selection import train_test_split

    Y=to_categorical(Y)

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

    np.random.seed(42)

    rn.seed(42)

    tf.compat.v1.set_random_seed(42)

    model=Sequential()
    model.add(Dense(input_dim=23,units=8,activation='relu'))
    model.add(Dense(units=16,activation='relu'))
    model.add(Dense(units=2,activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

    model.summary()

    print(x_train.shape)
    print(y_train)

    History=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,verbose=1)

    predict_x=model.predict(x_test) 
    classes_x=np.argmax(predict_x,axis=1)

    model.predict(x_test)

    model.evaluate(x_test,y_test)

    from tensorflow import keras
    model.save("attrition.model")

