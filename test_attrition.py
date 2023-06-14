def Testing_1(File):
    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns
    import missingno as msno
    import pickle
    import csv
    style.use('fivethirtyeight')
    sns.set(style='whitegrid',color_codes=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder

    f = open('model_select.txt', 'r')
    file1 = f.read()
    f.close()
    print(file1)
    
    model = pickle.load(open(file1, 'rb'))
    print("model_loaded")

    df=pd.read_csv(File)
    print(df.shape)
    print("starting {} ".format(df))
    print("data_readed")
    def transform(feature):
        le=LabelEncoder()
        df[feature]=le.fit_transform(df[feature])
        print(le.classes_)
    cat_df=df.select_dtypes(include='object')
    for col in cat_df.columns:
        transform(col)
    print("data_labeled_to numbers")
    print("second {} ".format(df))
    lbl=df["EmployeeNumber"]
    df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
            ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)
    print("dropped_unwanted_datas")
    print(len(df.columns))

    scaler=StandardScaler()
    scaled_df=scaler.fit_transform(df)
    X=scaled_df
    np.savetxt("reduced_test_data.csv",X, fmt='%s', delimiter=",")
    print("Saved the reduced features")

    Col = []
    with open('reduced_test_data.csv', 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            label =row[0]
            features = row[:]
            prediction = model.predict([features])[0]
            if prediction == 0:
                Col.append('No')
            else:
                Col.append('Yes')

    # print(List)
    result_file=pd.read_csv(File)
    result_file['Attrition'] = Col
    result_file.to_csv("result.csv",index=False)
