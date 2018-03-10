import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score



def preprocess(filename):
    """

    :return: Test and train splits after processing
    """
    # step 1: Load the data and include column headers
    dataframe_all = pd.read_csv('bcw.txt', sep=",", header=None)
    dataframe_all.columns = ["ID", "CT", "cell_size", "cell_shape", "MA", "ECS", "bare_nuclei", "BC", "normal_nuclei",
                             "Mitoses", "CLASS"]

    num_rows = dataframe_all.shape[0]

    # Total number of matches.
    n_matches = dataframe_all.shape[0]

    # Calculate number of features. -1 because we are saving one as the target variable (Benign or malignant)
    n_features = dataframe_all.shape[1] - 1

    # Print the results
    print("Total number of cases: {}".format(num_rows))
    print("Number of features: {}".format(n_features))

    columns = dataframe_all.columns
    print(columns)

    # step 2: remove useless data
    # count the number of missing elements (NaN) in each column
    counter_nan = dataframe_all.isnull().sum()
    counter_without_nan = counter_nan[counter_nan==0]
    # remove the columns with missing elements
    dataframe_all = dataframe_all[counter_without_nan.keys()]
    # remove the first column which contains no discriminative information (Sample IDs)
    dataframe_all = dataframe_all.drop(dataframe_all.columns[[0]], axis=1)
    # Modify CLASS labels from 2,4 to 0,1
    dataframe_all["CLASS"] = dataframe_all["CLASS"].apply(lambda x: x / 2 - 1)
    # the list of columns (the last column is the class label)
    columns = dataframe_all.columns
    print(columns)

    # step 3: get features (x) and scale the features
    # get x and convert it to numpy array
    x = dataframe_all.ix[:,:-1].values
    standard_scaler = StandardScaler()
    x_std = standard_scaler.fit_transform(x)

    # step 4: get class labels y and then encode it into number
    # get class label data
    y = dataframe_all.ix[:,-1].values
    # encode the class label
    class_labels = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # step 5: split the data into training set and test set
    test_percentage = 0.1
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

    return x_train, x_test, y_train, y_test,dataframe_all

