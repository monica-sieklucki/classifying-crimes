"#!/usr/bin/env python3"
import sys
import pandas as pd
from pandas import DataFrame

#pd.set_option('display.max_columns', None)


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
    df = extract_target_values(raw_df, column_name, relevant_target_values)  # filtering the raw dataset

    return df  # returning the filtered dataframe


def clean_dataset(data_frame):
    #Drop 'Case number'
    '''
    Removing all columns that do not provide any relevant insight into the data:
    '''
    data_frame = data_frame.drop(columns=["ID", "Case Number", "IUCR", "Beat", "District", "Community Area", "FBI Code", "X Coordinate", "Y Coordinate", "Year", "Updated On", "Latitude", "Longitude", "Location"], axis=1)

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
        updated_block_list.append(address[6:]) # split at 6th index, all the way till the end of the string
        i = i + 1  # increment the stupid counter

    # remove the current 'Block' column from the received data_frame
    data_frame = data_frame.drop(columns=["Block"], axis=1)

    # combine the processed_block_df to the data_frame
    data_frame['Block'] = updated_block_list

    return data_frame


def encode_block(data_frame):
    blocks_encoded = pd.get_dummies(data_frame.Block, prefix='Block')
    encoded_df = pd.merge(data_frame, blocks_encoded, left_index=True, right_index=True)
    return encoded_df


def encode_description(data_frame):
    description_encoded = pd.get_dummies(data_frame.Description, prefix='Description')
    encoded_df = pd.merge(data_frame, description_encoded, left_index=True, right_index=True)
    return encoded_df


def preprocess(data_frame):
    # remove unecessary address numbers
    data_frame = strip_block_numbers(data_frame)

    # on-hot-encode the Block column
    data_frame = encode_block(data_frame)

    # one-hot encode the Description column
    data_frame = encode_description(data_frame)

    return data_frame


def main():
    # Keeping name upper-cased to let the world know this var shouldn't be changed
    RAW_DATA_PATH = "data/filtered-crime-data.csv"

    df = filter_dataset(RAW_DATA_PATH)
    df = clean_dataset(df)

    # Perform any preprocessing logic
    df = preprocess(df)

    # store the filtered dataset (not necessary, but useful  JiC).
    df.to_csv(index=False, path_or_buf="data/filtered-crime-data.csv")

    print(df.head())


if __name__ == "__main__":
    main()
