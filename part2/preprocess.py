# coding: utf-8
import pandas as pd
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Impute the missing ages with median age
train["Age"] = train["Age"].fillna(train["Age"].median()).astype(int)
test["Age"] = test["Age"].fillna(test["Age"].median()).astype(int)

# Fill in missing embarked with S
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

# Fill in missing Cabin with None
train["Cabin"] = train["Cabin"].fillna("None")
test["Cabin"] = test["Cabin"].fillna("None")

# Write our changed dataframes to csv.
test.to_csv("./test.csv", index=False)
train.to_csv('./train.csv', index=False)
