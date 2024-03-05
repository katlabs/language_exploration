# Data class to load and manage cleaned dataset for Modeling and Analysis
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Data_Import():
    """ 
    Sets up dataset and basic splitting functions
    """
    
    def __init__(self, data="data/main_df.csv", limit_ngrams=None, drop_languages=None):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise Exception("Invalid data, must be string or DataFrame object", data)
        self.__encode_labels(self.df)
        self.split_data(limit_ngrams=limit_ngrams, drop_languages=drop_languages)
        self.n = self.X.shape[0]
    
    def __encode_labels(self, df):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(df['language'])
        self.y = self.le.transform(df['language'])
        
    def __get_features(self, limit_ngrams=None, drop_features=None, drop_languages=None):
        self.X = self.df
        if limit_ngrams != None:
            self.X = self.X.iloc[:, 0:limit_ngrams+5]
        if drop_features != None:
            self.X = self.X.drop(columns=drop_features)
        if drop_languages != None:
            for language in drop_languages:
                self.X = self.X.drop(self.X.loc[self.X['language']==language].index)
            self.X.reset_index(drop=True,inplace=True)
        self.__encode_labels(self.X)
        self.X = self.X.drop(["language","text"], axis=1)
        self.n = self.X.shape[0]
    
    def split_data(self, limit_ngrams=None, drop_features=None, drop_languages=None, test_size=0.25, random_state=42):
        self.__get_features(limit_ngrams=limit_ngrams, drop_features=drop_features, drop_languages=drop_languages)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)