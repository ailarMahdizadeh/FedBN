"""
This file is used to pre-process all data in Digit-5 dataset.
i.e., splitted data into train&test set  in a stratified way.
The function to process data into 10 partitions is also provided.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import pickle as pkl
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import  Counter
import requests
from tqdm import tqdm
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        total_length = response.headers.get('content-length')
    print('Downloading...')
    save_response_content(response, destination, total_length)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, total_length):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        total_length = int(total_length)
        for chunk in tqdm(response.iter_content(CHUNK_SIZE),total=int(total_length/CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
                

def stratified_split(X,y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Train:', Counter(y_train))
        print('Test:', Counter(y_test))

    return (X_train, y_train), (X_test, y_test)


def process_Caltech():
    # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Caltech/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Caltech/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Caltech/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Caltech/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Caltech/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Caltech/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_KKI():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/KKI/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/KKI/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/KKI/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/KKI/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/KKI/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/KKI/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_CMU():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/CMU/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/CMU/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/CMU/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/CMU/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/CMU/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/CMU/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))


def process_Leuven():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Leuven/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Leuven/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Leuven/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Leuven/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Leuven/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Leuven/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_NYU():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/NYU/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/NYU/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/NYU/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/NYU/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/NYU/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/NYU/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_MaxMun():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/MaxMun/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/MaxMun/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/MaxMun/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/MaxMun/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/MaxMun/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/MaxMun/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))


def process_OHSU():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/OHSU/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/OHSU/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/OHSU/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/OHSU/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/OHSU/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/OHSU/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_Olin():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Olin/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Olin/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Olin/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Olin/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Olin/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Olin/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_Pitt():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Pitt/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Pitt/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Pitt/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Pitt/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Pitt/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Pitt/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_SBL():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/SBL/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/SBL/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/SBL/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/SBL/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/SBL/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/SBL/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_SDSU():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/SDSU/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/SDSU/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/SDSU/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/SDSU/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/SDSU/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/SDSU/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_Stanford():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Stanford/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Stanford/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Stanford/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Stanford/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Stanford/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Stanford/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_Trinity():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Trinity/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Trinity/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Trinity/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Trinity/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Trinity/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Trinity/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_UCLA():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/UCLA/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/UCLA/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/UCLA/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/UCLA/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/UCLA/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/UCLA/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_UM():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/UM/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/UM/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/UM/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/UM/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/UM/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/UM/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_USM():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/USM/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/USM/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/USM/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/USM/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/USM/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/USM/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))

def process_Yale():
        # Initialize an empty DataFrame to store the subject information
    subject_info_df = pd.read_csv('../fc/subject_info.csv')
    X1_all = []
    X2_all = []
    Y_all = []
    mat_folder_x1 = '../../FedBN-master/data/FL_200/Yale/'
    mat_folder_x2 = '../../FedBN-master/data/FL_392/Yale/'

    for filename in os.listdir(mat_folder_x1):
        if filename.endswith(".mat"):
            # Extract the subject ID from the MAT file name
            subject_id = int(filename.split('.')[0].strip())
            
            # Find the corresponding DX_GROUP value in the DataFrame for x1
            dx_group_x1 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x1) > 0:
                print(f"x1 - Subject ID {subject_id}: DX_GROUP {dx_group_x1[0]}")
                
                # Modify the DX_GROUP value based on your criteria for x1
                if dx_group_x1[0] == 1:
                    dx_group_value_x1 = 0
                elif dx_group_x1[0] == 2:
                    dx_group_value_x1 = 1
                else:
                    dx_group_value_x1 = dx_group_x1[0]  # Keep the value unchanged
                
                print(f"x1 - Modified DX_GROUP value: {dx_group_value_x1}")
                
                # Load your data from the MAT file for x1
                data_x1 = scipy.io.loadmat(os.path.join(mat_folder_x1, filename))
                
                # Append data to X1_all and modified DX_GROUP value to Y_all
                X1_all.append(data_x1)
                Y_all.append(dx_group_value_x1)
            else:
                print(f"x1 - Subject ID {subject_id} not found in the subjectinfo.csv file.")

    # Split the data into training and testing sets for x1
    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_all, Y_all, test_size=0.2, random_state=42)

    # Now, X1_train contains your x1 data from all the processed MAT files, and Y_train contains
    # the modified DX_GROUP values corresponding to the x1 data.

    # Load x2 data using the same test split indices as x1
    for filename in os.listdir(mat_folder_x2):
        if filename.endswith(".mat"):
            subject_id = int(filename.split('.')[0].strip())
            
            dx_group_x2 = subject_info_df.loc[subject_info_df['SUB_ID'] == subject_id, 'DX_GROUP'].values
            if len(dx_group_x2) > 0:
                print(f"x2 - Subject ID {subject_id}: DX_GROUP {dx_group_x2[0]}")
                
                if dx_group_x2[0] == 1:
                    dx_group_value_x2 = 0
                elif dx_group_x2[0] == 2:
                    dx_group_value_x2 = 1
                else:
                    dx_group_value_x2 = dx_group_x2[0]
                
                print(f"x2 - Modified DX_GROUP value: {dx_group_value_x2}")
                
                data_x2 = scipy.io.loadmat(os.path.join(mat_folder_x2, filename))
                
                # Append data to X2_all (use the same test split indices as x1)
                X2_all.append(data_x2)

    # Split the x2 data into training and testing sets using the same indices as x1_test
    X2_train, X2_test = train_test_split(X2_all, test_size=len(X1_test), random_state=42)

    # Now, X2_train contains your x2 training data, and X2_test contains your x2 testing data,
    # both using the same test split indices as x1_test.

    # Save x1 training and testing data with labels
    with open('../../FedBN-master/data/FL_200/Yale/train_x1.pkl', 'wb') as f:
        pickle.dump((X1_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_200/Yale/test_x1.pkl', 'wb') as f:
        pickle.dump((X1_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Save x2 training and testing data with labels
    with open('../../FedBN-master/data/FL_392/Yale/train_x2.pkl', 'wb') as f:
        pickle.dump((X2_train, Y_train), f, pickle.HIGHEST_PROTOCOL)

    with open('../../FedBN-master/data/FL_392/Yale/test_x2.pkl', 'wb') as f:
        pickle.dump((X2_test, Y_test), f, pickle.HIGHEST_PROTOCOL)

    # Print the shapes of the training and testing data and labels
    print('Train x1 shape:\t', len(X1_train))
    print('Train x2 shape:\t', len(X2_train))
    print('Test x1 shape:\t', len(X1_test))
    print('Test x2 shape:\t', len(X2_test))
    print('Train labels:\t', len(Y_train))
    print('Test labels:\t', len(Y_test))
