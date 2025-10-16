import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def run_knn_analysis(datasets, random_state=42):
    # Process Dataset 1
    if "Dataset1" in datasets:
        print("\n------- Processing Dataset 1 -------")
        scaler = MinMaxScaler()
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

        project1_dataset1 = pd.read_csv(datasets["Dataset1"], header=None, sep="\t")
        project1_dataset1 = pd.DataFrame(
            scaler.fit_transform(project1_dataset1), columns=project1_dataset1.columns
        )
        X1 = project1_dataset1.drop(30, axis=1)
        y1 = project1_dataset1[30]
        knn1 = KNeighborsClassifier(
            n_neighbors=13
        )  # k=13 seems to yield the best accuracy score for this model
        knn1.fit(X1, y1)
        y1_pred = knn1.predict(X1)

        print("\n----- Model 1 scores (k=13) -----")
        score = cross_val_score(knn1, X1, y1, cv=kf, scoring="accuracy")
        print(f'Average accuracy score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn1, X1, y1, cv=kf, scoring="precision")
        print(f'Average precision score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn1, X1, y1, cv=kf, scoring="recall")
        print(f'Average recall score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn1, X1, y1, cv=kf, scoring="f1")
        print(f'Average f1 score: {"{:.4f}".format(score.mean())}')

    # Process Dataset 2
    if "Dataset2" in datasets:
        print("\n------- Processing Dataset 2 -------")
        scaler = MinMaxScaler()
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

        project1_dataset2 = pd.read_csv(datasets["Dataset2"], header=None, sep="\t")
        project1_dataset2 = pd.get_dummies(
            project1_dataset2, columns=[4], drop_first=True
        )
        project1_dataset2.rename(columns={"4_Present": 4}, inplace=True)
        project1_dataset2 = pd.DataFrame(
            scaler.fit_transform(project1_dataset2), columns=project1_dataset2.columns
        )
        X2 = project1_dataset2.drop(9, axis=1)
        y2 = project1_dataset2[9]
        knn2 = KNeighborsClassifier(
            n_neighbors=27
        )  # k=27 seems to yield the best accuracy score for this model
        knn2.fit(X2, y2)
        y2_pred = knn2.predict(X2)

        print("\n----- Model 2 scores (k=27) -----")
        score = cross_val_score(knn2, X2, y2, cv=kf, scoring="accuracy")
        print(f'Average accuracy score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn2, X2, y2, cv=kf, scoring="precision")
        print(f'Average precision score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn2, X2, y2, cv=kf, scoring="recall")
        print(f'Average recall score: {"{:.4f}".format(score.mean())}')
        score = cross_val_score(knn2, X2, y2, cv=kf, scoring="f1")
        print(f'Average f1 score: {"{:.4f}".format(score.mean())}')


## Test code for finding the best value of k
# print('\n')
# max_accuracy, k_max = 0, 0
# for k in range(1, 31):
#     knn2 = KNeighborsClassifier(n_neighbors=k)
#     knn2.fit(X2, y2)
#     score = cross_val_score(knn2, X2, y2, cv=kf, scoring='accuracy')
#     if score.mean() > max_accuracy:
#         max_accuracy = score.mean()
#         k_max = k
#     print(f'Average accuracy score (k = {k}): {"{:.4f}".format(score.mean())}')
# print(f'Max accuracy score: {max_accuracy:.4f} (k = {k_max})')
