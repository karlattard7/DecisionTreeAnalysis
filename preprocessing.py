import pandas as pd


def readParseDataset(file_name, column_labels):
    """ Function that reads and parses CSV files """

    location = 'Dataset/'  # directory where datasets used are placed

    df = pd.read_csv(location + file_name)  # reading CSV file
    df.columns = column_labels

    # Target column must always be last column - to make code more general
    if df.columns[len(column_labels)-1] != 'Class':
        df = changeTargetColumnLoc(df)

    # continuous-valued features
    df_continuous_columns = list(df.select_dtypes(include=['float', 'int']).columns)

    return convertContinuousToCategorical(df, df_continuous_columns)


def changeTargetColumnLoc(df):
    """ Changing position of target column, i.e. 'Class' """
    target_class = df['Class'].to_numpy()
    df = df.drop('Class', axis=1)
    df['Class'] = target_class

    return df


def splitTrainTest(df):
    """ Function that splits dataframe into train and test sets """

    df = df.sample(frac=1).reset_index(drop=True)  # Shuffling df and resetting indexes

    # df will be split 80% Training Set & 20% Test Set
    train_len = int(len(df) * 0.8)

    return df[:train_len], df[train_len:].reset_index(drop=True)


def convertContinuousToCategorical(df, columns):
    """ Function that converts dataset's continuous variables to categorical variables """

    if columns is None:  # no continuous attributes to convert
        return df

    # Iterating over every continuous-valued features
    # Generating Q1 and Q3 values for feature currently iterating in
    # Converting continuous values to categorical based on thresholds (Q1 and Q3)
    for column in columns:
        df[column] = df[column].astype(int)
        if column != 'Class':
            # Calculating Thresholds - thresholds are Q1 and Q3
            first_quartile = int(df[column].describe()['25%'])  # Q1 value
            third_quartile = int(df[column].describe()['75%'])  # Q3 value

            # Converting to str - utility purposes
            first_quartile_str = str(first_quartile)
            third_quartile_str = str(third_quartile)

            # Converting Continuous values to categorical values based on the thresholds calculated
            df.loc[(df[column] >= 0)
                   & (df[column] < first_quartile), column + '_cat'] = column + '<' + first_quartile_str

            df.loc[(df[column] >= first_quartile) &
                   (df[column] < third_quartile), column + '_cat'] = column + '<' + third_quartile_str

            df.loc[df[column] >= third_quartile, column + '_cat'] = column + '>=' + third_quartile_str

    df = df.drop(columns[:-1], axis=1)

    return changeTargetColumnLoc(df)


def convertNoneTypesToUnknowns(predictions):
    """ This function is used to convert None to 'Unknown' (For evaluation purposes) """

    for i in range(len(predictions)):
        if predictions[i] is None:
            predictions[i] = "unknown"

    return predictions
