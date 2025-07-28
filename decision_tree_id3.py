import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from preprocessing import convertNoneTypesToUnknowns
eps = np.finfo(float).eps  # this is done to avoid 0 in the denominator or log(0)

""" 
A Decision Tree Classifier using ID3 is built in the following way:

1. Computing the Entropy and Information Gain for the training set

2. Then, calculate the Entropy for each attribute in the training set

3. Compute the Information Gain for each attribute

4. Choose the attribute which has the highest IG and create a node for it

5. The above is repeated until desired tree is built

"""


def calculateDatasetEntropy(df):
    """ This function calculates the entropy of whole dataset """

    target_class = df.columns[-1]  # taking last column, i.e. target class
    total_entropy = 0  # holds entropy of dataset
    class_values = df[target_class].unique()

    # Iterating over every class value and computing its entropy
    for value in class_values:
        proportion = df[target_class].value_counts()[value] / len(df[target_class])
        class_entropy = - (proportion * np.log2(proportion))  # computing the respective class entropy
        total_entropy += class_entropy  # summation of all classes entropy's

    return total_entropy


def calculateAttributeEntropy(df, attribute):
    """ This function calculates the entropy of an attribute """

    target_class = df.columns[-1]
    target_class_values = df[target_class].unique()  # Getting all unique target classes
    attribute_values = df[attribute].unique()  # Getting all attribute unique values
    attribute_entropy = 0  # holds entropy of attribute

    # Nested for loop that iterates through attribute's values.
    # For each attribute value, it iterates over every target class value and computes its entropy.
    # Nested inner loop sums up all the entropy's of current attribute's value respective target class value
    # Outer loop sums up all the entropy's of each attribute's values
    for attribute_value in attribute_values:
        total_attribute_value_entropy = 0  # holds the total entropy of current attribute value

        for class_value in target_class_values:
            value1 = len(df[attribute][df[attribute] == attribute_value][df[target_class] == class_value])
            value2 = len(df[attribute][df[attribute] == attribute_value]) + eps
            proportion = value1 / value2

            # computing the entropy the current attribute value at the current respective class
            attribute_value_class_entropy = - (proportion * np.log2(proportion + eps))

            # Summation of the entropy of each target class in the respective attribute value
            total_attribute_value_entropy += attribute_value_class_entropy

        final_value = - ((value2 / len(df)) * total_attribute_value_entropy)
        attribute_entropy += final_value  # Summation of all value's entropy for the attribute passed as a parameter

    attribute_entropy = abs(attribute_entropy)

    return attribute_entropy


def buildDecisionTreeID3(df, tree=None):
    """ This function recursively recalls itself to build the Decision Tree using ID3 """

    target_class = df.columns[-1]

    # list of all information gains of each attribute
    information_gain_list = [calculateDatasetEntropy(df) - calculateAttributeEntropy(df, column) for column in
                             df.columns[:-1]]

    # Choosing column having highest information gain to be the current node
    index_highest_ig = np.argmax(information_gain_list)
    node = df.columns[:-1][index_highest_ig]

    # Getting all unique values of chosen attribute
    attribute_values_unique = np.unique(df[node])

    # The Decision Tree built is a dictionary
    if tree is None:
        tree = {node: {}}

    # Iterating over every unique attribute value
    for value in attribute_values_unique:

        sub_tree = df[df[node] == value].reset_index(drop=True)
        val, counts = np.unique(sub_tree[target_class], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = val[0]
        else:
            tree[node][value] = buildDecisionTreeID3(sub_tree)  # recursive call

    return tree


def predict(tree, instance):
    """ This function predicts unseen data """

    if not isinstance(tree, dict):  # if it is leaf node
        return tree  # return the value

    else:
        # Sub-Tree, i.e. continue iterating

        root_node = next(iter(tree))  # getting first key/feature name of the dictionary
        feature_value = instance[root_node]  # value of the feature
        
        if feature_value in tree[root_node]:  # checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance)  # goto next feature
        else:
            return None


def getPredictedValues(tree, test_set):
    """ This function returns all the predicted outputs """

    predictions = []  # holds all predicted classes
    for index, row in test_set.iterrows():  # for each row in the dataset
        predictions.append(predict(tree, test_set.iloc[index]))  # predict the row

    return predictions


def classificationReport(tree, df_test, dataset_name, to_print=True):
    """ Evaluates the Built Decision Tree Classifier - Classification Report """

    y_true = df_test['Class'].to_list()
    y_pred = getPredictedValues(tree, df_test)
    y_pred = convertNoneTypesToUnknowns(y_pred)

    df = pd.DataFrame(classification_report(y_true, y_pred, labels=np.unique(y_pred), output_dict=True))  # cr -> df
    columns = list(df.columns[:-2])  # needed columns

    # Removing 'unknown' -> this appear when the model predicted None, i.e. no path existed in trained decision tree
    if 'unknown' in columns:
        columns.remove('unknown')

    # Getting Accuracy from df & Dropping unneeded columns
    accuracy = round(df['accuracy'].iloc[0] * 100, 2)
    df = df[columns].drop('support', axis=0).drop('accuracy', axis=1)

    df = df.apply(lambda x: round(x * 100, 2))  # Converting decimals to percentages

    if to_print:
        print(f"\nClassification Report for {dataset_name} Evaluation Dataset")
        print("---------------------------------------------------------------")
        print(df)

        print(f"Overall Accuracy is {accuracy}%")

    return accuracy


def postPruning(tree, train_df, test_df):
    """ This function Performs Post Pruning Overfitting Management Algorithm on the Specified Trained Tree"""

    key_feature = list(tree.keys())[0]  # holds current tree key

    # Holds a list of boolean values: True -> key's value is a tree; False -> Value is not a tree
    has_subtree = [isinstance(tree[key_feature][value], dict)
                   for value in train_df[key_feature] if value in tree[key_feature].keys()]

    if True not in has_subtree:
        return prune(tree, train_df, test_df)  # Pruning sub-tree

    else:
        temp = {}
        for value in tree[key_feature]:
            if isinstance(tree[key_feature][value], dict):
                # Recursive call - Reaching leaf node
                temp[value] = postPruning(tree[key_feature][value], train_df[train_df[key_feature] == value],
                                          test_df[test_df[key_feature] == value])

            else:
                temp[value] = tree[key_feature][value]  # Storing leaf node

        tree = {key_feature: temp}  # constructing the pruned tree bottom-up

        return tree


def prune(tree, train_df, test_df):
    """ Pruning the passed sub-tree """

    potential_leaf = train_df['Class'].value_counts().index[0]  # getting the most occurred target column value

    errors_pruned_leaf = sum(test_df['Class'] != potential_leaf)  # Getting errors on leaf node
    errors_sub_tree = evaluateTreeOnTest(tree, test_df)  # getting errors on current sub-tree

    # If accuracy did not improve, then keep current sub-tree. Otherwise, prune, hence return leaf node
    # Comparing if error in pruned tree is higher than error in subtree
    if errors_pruned_leaf > errors_sub_tree:
        return tree
    else:
        return potential_leaf


def evaluateTreeOnTest(tree, test_df):
    """ Returning the number of good prediction of passed tree """

    queries = test_df.drop('Class', 1).to_dict(orient='records')

    # Predicting on test data
    temp_predicted = [predict(tree, queries[i]) for i in range(len(test_df))]

    df = pd.DataFrame(temp_predicted, columns=['Predictions'])

    return np.sum(df['Predictions'].equals(test_df['Class']))  # Returns the sum of good predictions
