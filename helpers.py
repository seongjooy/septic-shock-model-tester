import feature_lists
import pandas as pd
import joblib
from sklearn.utils import resample


def get_map(dataset, num):
    return (2 * dataset[f'dbp_{num}'] + dataset[f'sbp_{num}']) / 3


def get_si(dataset, num):
    return dataset[f'hr_{num}'] / dataset[f'sbp_{num}']


def feature_adder(train_set, test_set, num_features, col1, col2):
    # return train_set, test_set
    pass


# function adding SI and MAP
# input: two test sets
# output: two test sets with SI and MAP, hard-coded for shock data
def si_map_adder(train_set, test_set):
    for i in range(1, 7):
        train_set[f'map_{i}'] = get_map(train_set, i)
        test_set[f'map_{i}'] = get_map(test_set, i)

    for i in range(1, 7):
        train_set[f'si_{i}'] = get_si(train_set, i)
        test_set[f'si_{i}'] = get_si(test_set, i)

    return train_set, test_set


# input: two datasets, int hours
# output: two datasets with hours filtered as desired
def hour_selector(train_set, test_set, hours):
    hours_to_drop = (set(feature_lists.get_hour_list()) - set(int(hour) for hour in hours))
    to_drop = [feature + str(num_hour) for feature in feature_lists.get_feature_names() for num_hour in hours_to_drop]

    return train_set.drop(to_drop, axis=1), test_set.drop(to_drop, axis=1)


# input: two datasets, column label
# output: two datasets resized to 1:1 ratio, specific to binary classification datasets, scale to pos
def downsample_datasets_one_to_one(d1, d2, target_var):
    d1_neg, d1_pos = d1[d1[target_var] == 0], d1[d1[target_var] == 1]
    d2_neg, d2_pos = d2[d2[target_var] == 0], d2[d2[target_var] == 1]

    # check condition if d1_neg < d1_pos for other datasets
    d1_neg_downsampled = resample(d1_neg,
                                  n_samples=len(d1_pos),
                                  random_state=123)
    d2_neg_downsampled = resample(d2_neg,
                                  n_samples=len(d2_pos),
                                  random_state=123)
    return pd.concat([d1_pos, d1_neg_downsampled]), pd.concat([d2_pos, d2_neg_downsampled])


def save_model_valid_input(value):
    if value == 'y':
        pass
    if value == 'n':
        pass
    else:
        save_model_valid_input(input('Invalid Input.\n Save model as a joblib file? y/n\n'))


# ask for input and check for error
def save_model(model, value):
    if value == 'y':
        joblib.dump(model, 'xgb_model.joblib')


