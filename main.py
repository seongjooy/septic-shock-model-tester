import time
import pandas as pd
import xgboost as xgb
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import helpers
import feature_lists
import joblib

pd.options.display.max_columns = None

# read csv files
train = pd.read_csv('training_7hours_shock.csv').drop(feature_lists.get_onset_times(), axis=1)
test = pd.read_csv('test_7hours_shock.csv').drop(feature_lists.get_onset_times(), axis=1)

# adding SI and MAP
train, test = helpers.si_map_adder(train, test)

# asking for hour input, this is input to the helper function hour_selector
hours = input('Data is available for hours 1~6.\n'
              'Enter hours to use: \n'
              'ex. To select hours 2,3,4, enter 2 3 4.\n'
              ).split(' ')

# selecting hours
train, test = helpers.hour_selector(train, test, hours)

# downsampling datasets
train_balanced, test_balanced = helpers.downsample_datasets_one_to_one(train, test, 'dlabel')

print(f'Using hours {[int(hour) for hour in hours]}')
# time.sleep(1)

save = helpers.save_model_valid_input(input('Save model as joblib file? y/n\n'))


print(train_balanced.columns, test_balanced.columns)
# time.sleep(1)

# 5-fold validation
skFold = StratifiedKFold(n_splits=5,
                         random_state=123, shuffle=True)

X_train_kf, y_train_kf = train_balanced.drop(['dlabel'], axis=1), \
                         train_balanced['dlabel']

X_test, y_test = test_balanced.drop(['dlabel'], axis=1), \
                 test_balanced['dlabel']

best_idx = best_AUC = 0
best_report = report = None
xgb_model = None

for idx, (train_idx, val_idx) in enumerate(skFold.split(X_train_kf, y_train_kf)):
    X_train, X_val = X_train_kf.iloc[train_idx, :], X_train_kf.iloc[val_idx, :]
    y_train, y_val = y_train_kf.iloc[train_idx], y_train_kf.iloc[val_idx]
    eval_set = [(X_val, y_val)]

    xgb_model = xgb.XGBClassifier(
        seed=123,
        use_label_encoder=False,
        learning_rate=0.01,
        max_depth=4,
        n_estimators=2000,
        subsample=0.25,
        colsample_bytree=0.6,
        eval_metric='aucpr',
        verbosity=1,
        scale_pos_weight=2.5,
        gamma=8
    )

    xgb_model.fit(
        X_train,
        y_train,
        early_stopping_rounds=30,
        eval_set=eval_set,
        verbose=False
    )

    y_test_pred = xgb_model.predict_proba(X_test)[:, 1]
    curr_AUC = sklearn.metrics.roc_auc_score(y_test, y_test_pred)

    print(f'\nCV Set {idx + 1}:\n'
          f'Test AUC: {curr_AUC}')

    y_pred = xgb_model.predict(X_test)

    predictions = [value for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    report = sklearn.metrics.classification_report(y_test, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    print(f'Test Accuracy: {accuracy}\n'
          f'{report}\n'
          f'{confusion_matrix}\n')

    if curr_AUC > best_AUC:
        best_AUC = curr_AUC
        best_idx = idx
        best_report = report

print(f'Best set by AUC: {best_idx + 1}\n'
      f'AUC: {round(best_AUC, 3)}\n'
      f'{best_report}')


helpers.save_model(xgb_model, save)
