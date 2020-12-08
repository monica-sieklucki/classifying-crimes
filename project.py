"""#!/usr/bin/env python3"""
import sys
import pandas as pd
import numpy as np
import sns as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def extract_target_values(df, column_name, relevant_target_values):
    """
    :param df: The pandas dataframe to be extracted from.
    :param column_name: The name of the column to filter values by.
    :param relevant_target_values: The relevant values to include in the
           filtered dataset.
    :return: A pandas dataframe containing only rows where the `column_name`
             matches any one of the provided `relevant_target_values`.
    """
    return df.loc[df[column_name].isin(relevant_target_values)]


def filter_dataset(raw_data_path):
    """
    filters the provided raw dataset.
    :param raw_data_path: path to the raw, unprocessed dataset.
    :return: a filtered pandas dataframe.
    """
    raw_dataset_filepath = "data/unprocessed-crime-data.csv"  # raw, unprocessed dataset location.
    raw_df = pd.read_csv(raw_dataset_filepath)  # loading the raw data.
    column_name = "Primary Type"  # the column to filter on
    relevant_target_values = ["BATTERY", "THEFT", "CRIMINAL DAMAGE"]  # can be changed for experimentation
    # "ROBBERY", "ASSAULT", "WEAPONS", "MOTOR VEHICLE THEFT"
    df = extract_target_values(raw_df, column_name, relevant_target_values)  # filtering the raw dataset

    return df  # returning the filtered dataframe


def clean_dataset(data_frame):
    # Drop 'Case number'
    """
    Removing all columns that do not provide any relevant insight into the data:
    """
    data_frame = data_frame.drop(
        columns=["ID", "Case Number", "IUCR", "Beat", "District", "Community Area", "FBI Code", "X Coordinate",
                 "Y Coordinate", "Year", "Updated On", "Latitude", "Longitude", "Location", "Date"], axis=1)
    return data_frame


def strip_block_numbers(data_frame):
    # extract the 'Block' column
    block_df = data_frame["Block"]

    # convert the one-dimension df to a list
    block_list = block_df.values.tolist()

    # iterate through each item in the list and remove the first 6 elements of the string (the numeric address portion)
    i = 0
    updated_block_list = []

    for address in block_list:
        updated_block_list.append(address[6:])  # split at 6th index, all the way till the end of the string
        i = i + 1  # increment the stupid counter

    # remove the current 'Block' column from the received data_frame
    data_frame = data_frame.drop(columns=["Block"], axis=1)

    # combine the processed_block_df to the data_frame
    data_frame['Block'] = updated_block_list

    return data_frame


def encode_column(data_frame, column_name):
    column_encoded = pd.get_dummies(data_frame[column_name], prefix=column_name)
    # remove the old `Location Description` column from the received data_frame
    data_frame = data_frame.drop(columns=[column_name], axis=1)
    encoded_df = pd.merge(data_frame, column_encoded, left_index=True, right_index=True)
    return encoded_df


def encode(data_frame):
    # remove unnecessary address numbers
    data_frame = strip_block_numbers(data_frame)

    # on-hot-encode
    column_names = ["Description", "Location Description", "Arrest", "Domestic", "Ward", "Block"]

    for column_name in column_names:
        data_frame = encode_column(data_frame, column_name)

    return data_frame


'''
Creates a baseline Dummy Model using Sklearn's DummyClassifier. The value for the
DummyClassifier's 'strategy' parameter argument is passed in as an argument for this
method.
'''


def predict_with_baseline_dummy_model(_strategy: str, X: np.array, y: np.array, x: np.array) -> None:
    dummy_clf_stratified = DummyClassifier(strategy=_strategy)
    dummy_clf_stratified.fit(X, y)
    dummy_clf_stratified.predict(x)
    print('dummy_clf_stratified score (strategy={}): {}%'.format(_strategy, dummy_clf_stratified.score(X, y) * 100))


'''
Will split the specified dataset into three subsets -- training (70%),
develop (15%), and testing (15%).
'''


def split_into_train_develop_test(unlabeled_dataset: pd.DataFrame, dataset_label_values: pd.DataFrame):
    """
    Splitting dataset into a training (70%), development (15%) and test (15%).
    [NOTE: splitting develop and test into 30% for now. This 30% will be halved,
    yielding 15% and 15% of the total dataset, respectively]
    """
    X_train, X_develop_test, y_train, y_develop_test = train_test_split(unlabeled_dataset,
                                                                        dataset_label_values, test_size=0.30,
                                                                        random_state=42)

    '''
    Splitting the X_develop_test and y_develop_test (which are 30% of the dataset)
    in half, so that the resulting dataframe sizes yield 15% of the total dataset
    each.
    '''
    X_develop, X_test, y_develop, y_test = train_test_split(X_develop_test, y_develop_test,
                                                            test_size=0.5, random_state=42)
    return X_train, X_develop, X_test, y_train, y_develop, y_test


'''
Takes the entire labeled dataset, and "un-labels" it It simply removes the label from each
entry so that we can use the yielded unlabeled data to fit a Machine Learning model.
'''


def extract_unlabeled_data(dataset: pd.DataFrame, target_label: str) -> pd.DataFrame:
    return dataset.drop(target_label,
                        axis=1)  # axis=1 specifies that the deleted item is a column, as opposed to a row (0)


'''
Extracts a new DataFrame containing only the labels for each row.  This will be used to
compare the Machine Learning model's predictions with the actual values.
'''


def extract_target_label_values(dataset: pd.DataFrame, target_label: str) -> pd.DataFrame:
    return dataset[target_label]


def main():
    # Keeping name upper-cased to let the world know this var shouldn't be changed
    RAW_DATA_PATH = "data/filtered-crime-data.csv"

    df = filter_dataset(RAW_DATA_PATH)
    df = clean_dataset(df)

    # Perform any preprocessing logic
    df = encode(df)

    print("columns: {}".format(df.columns))

    # store the filtered dataset (not necessary, but useful  JiC).
    df.to_csv(index=False, path_or_buf="data/filtered-crime-data.csv")

    unlabeled_crime_dataset = extract_unlabeled_data(df.head(10000), "Primary Type")
    crime_dataset_label_values = extract_target_label_values(df.head(10000), "Primary Type")

    # Split data into train, develop, and test subsets for training/testing the baseline model
    X_train, X_develop, X_test, y_train, y_develop, y_test = split_into_train_develop_test(unlabeled_crime_dataset,
                                                                                           crime_dataset_label_values)

    # Establishing a baseline performance using Sklearn's DummyClassifier with strategy=stratified and most_frequent:
    predict_with_baseline_dummy_model("stratified", X_train, y_train, X_develop)
    predict_with_baseline_dummy_model("most_frequent", X_train, y_train, X_develop)

    # Performing fitting. Splitting dataset for training ad testing the actual model
    training_data, testing_data, training_labels, testing_labels = train_test_split(unlabeled_crime_dataset,
                                                                                    crime_dataset_label_values,
                                                                                    test_size=0.33, random_state=42)


    # plt.figure(figsize=(15, 10))
    # df_loc = df['Location Description'].value_counts().iloc[:20].index
    # sns.countplot(y='Location Description', data=df, order=df_loc)
    # plt.title('Top Description Locations')




    # teaching the model
    model = KNeighborsClassifier(n_neighbors=10, weights='uniform')  # LogisticRegression()
    model.fit(training_data, training_labels)


    logreg = LogisticRegression()
    logreg.fit(training_data, training_labels)


    svc_model = SVC()
    svc_model.fit(training_data, training_labels)

    decision_model = DecisionTreeClassifier()
    decision_model.fit(training_data, training_labels)



    param1 = {'max_depth': [10, 11, 13, 15], 'n_estimators': [100, 500, 1000]}
    param = dict(
        max_depth=[n for n in range(5, 15)],
        min_samples_split=[n for n in range(2, 6)],
        min_samples_leaf=[n for n in range(2, 5)],
        n_estimators=[50, 100],
        # criterion = ['gini','entropy']
    )
    grid_search_model = GridSearchCV(RandomForestClassifier(), param, cv=7)
    grid_search_model.fit(training_data, training_labels)

    # print("Best parameters {}".format(grid_search_model.best_params_) * 100)
    # print("Best score {:.4f}".format(grid_search_model.best_score_) * 100)

    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(training_data, training_labels)
    # print('Score = ', random_forest.score(training_data, training_labels))


    # make predictions
    kNN_predictions = model.predict(testing_data)

    logreg_predictions = logreg.predict(testing_data)

    random_forest_predictions = forest_model.predict(testing_data)

    svc_predictions = svc_model.predict(testing_data)

    decision_tree_predictions = decision_model.predict(testing_data)

    grid_search_predictions = grid_search_model.predict(testing_data)

    # gauging how good our model performed
    print('---------------------------------- K-Nearest Neighbors Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, kNN_predictions))
    print()
    print()
    print('---------------------------------- Logistic Regression Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, logreg_predictions))
    print()
    print()
    print('---------------------------------- Random Forest Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, random_forest_predictions))
    print()
    print()
    print('---------------------------------- SVC Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, svc_predictions))
    print()
    print()
    print('---------------------------------- Decision Tree Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, decision_tree_predictions))
    print()
    print()
    print('---------------------------------- Grid Search Prediction Accuracy: -----------------------------')
    print('Below are performance metrics.')
    print(classification_report(testing_labels, grid_search_predictions))


if __name__ == "__main__":
    main()
