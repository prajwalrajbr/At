
def Traning_1(File):
    # Ignore  the warnings(DeprecationWarning,RuntimeWarning)
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


    df=pd.read_csv(File)
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
            ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)

    def transform(feature):
        le=LabelEncoder()
        df[feature]=le.fit_transform(df[feature])
        print(le.classes_)

    cat_df=df.select_dtypes(include='object')
    cat_df.columns

    for col in cat_df.columns:
        transform(col)

    scaler=StandardScaler()
    scaled_df=scaler.fit_transform(df.drop('Attrition',axis=1))
    X=scaled_df
    Y=df['Attrition']

    """## 4.3 ) Splitting the data into training and validation sets"""

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)



    oversample=SMOTE()
    # oversample = SMOTE()
    # X, y = oversample.fit_resample(X, y)
    x_train_smote,  y_train_smote = oversample.fit_resample(x_train,y_train)
    import pickle

    def compare1(model):
        clf=model
        clf.fit(x_train_smote,y_train_smote)
        pred=clf.predict(x_test)
        
        # Calculating various metrics


        with open('boost.pkl', 'wb') as file:
            pickle.dump(clf,file)


        acc.append(accuracy_score(pred,y_test))
        prec.append(precision_score(pred,y_test))
        rec.append(recall_score(pred,y_test))
        auroc.append(roc_auc_score(pred,y_test))

    def compare2(model):
        clf=model
        clf.fit(x_train_smote,y_train_smote)
        pred=clf.predict(x_test)
        
        # Calculating various metrics

        with open('rf.pkl', 'wb') as file:
            pickle.dump(clf,file)


        acc.append(accuracy_score(pred,y_test))
        prec.append(precision_score(pred,y_test))
        rec.append(recall_score(pred,y_test))
        auroc.append(roc_auc_score(pred,y_test))

    def compare3(model):
        clf=model
        clf.fit(x_train_smote,y_train_smote)
        pred=clf.predict(x_test)
        
        # Calculating various metrics


        with open('svm.pkl', 'wb') as file:
            pickle.dump(clf,file)

        acc.append(accuracy_score(pred,y_test))
        prec.append(precision_score(pred,y_test))
        rec.append(recall_score(pred,y_test))
        auroc.append(roc_auc_score(pred,y_test))


    acc=[]
    prec=[]
    rec=[]
    auroc=[]
    models=[GradientBoostingClassifier() ,RandomForestClassifier(),SVC(kernel='rbf')]
    #(Radial Basis Function) kernel in SVM for non-linear classification allows the SVM to capture non-linear patterns
    #uses the distance between data points in a feature space to determine the similarity between them
    model_names=['Gradient Boosting Classifier (boost.pkl)','Random Forest Classifier(rf.pkl)','Support Vector Machine(svm.pkl)']

    ##for model in range(len(models)):
    compare1(models[0])
    compare2(models[1])
    compare3(models[2])
        
    d={'Modelling Algo':model_names,'Accuracy':acc,'Precision':prec,'Recall':rec,'Area Under ROC Curve':auroc}
    met_df=pd.DataFrame(d)
    met_df

    """## 5.3 ) Comparing Different Models"""

    def comp_models(met_df,metric):
        sns.factorplot(data=met_df,x=metric,y='Modelling Algo',size=5,aspect=1.5,kind='bar')
        sns.factorplot(data=met_df,y=metric,x='Modelling Algo',size=7,aspect=2,kind='point')

    comp_models(met_df,'Accuracy')
    plt.savefig("accuracy.jpg")

    comp_models(met_df,'Precision')
    plt.savefig("Precision.jpg")

    comp_models(met_df,'Recall')
    plt.savefig("Recall.jpg")

    comp_models(met_df,'Area Under ROC Curve')
    plt.savefig("ROC_curve.jpg")


    print("{} \n {}".format(x_train_smote.shape,y_train_smote.shape))

    return model_names, acc

