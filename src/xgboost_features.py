import gc

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)


gc.enable()

transactions = pd.read_csv('../input/processed_transaction_features.csv', index_col=0)

members = pd.read_csv('../input/members_v3.csv')

user_log_all = pd.read_csv('../input/processed_user_log_all.csv')
# user_log_test = pd.read_csv('../input/processed_features_user_log_all_time_including_mar.csv')
user_log_feb = pd.read_csv('../input/processed_features_user_log_feb.csv')
user_log_mar = pd.read_csv('../input/processed_features_user_log_mar.csv')

train = pd.read_csv('../input/train.csv')
train = train.append(pd.read_csv('../input/train_v2.csv'), ignore_index=True)

test = pd.read_csv('../input/sample_submission_v2.csv')

# Merge Data

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_log_all, how='left', on='msno')
test = pd.merge(test, user_log_all, how='left', on='msno')

train = pd.merge(train, user_log_feb, how='left', on='msno')
test = pd.merge(test, user_log_mar, how='left', on='msno')

train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

del transactions, members
gc.collect()

# Drop duplicates first
test = test.drop_duplicates('msno')

gender = {'male': 1, 'female': 2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train['bd'] = train['bd'].replace(0, train['bd'].mode())
test['bd'] = test['bd'].replace(0, test['bd'].mode())

train['gender'] = train['gender'].replace(0, train['gender'].mean())
test['gender'] = test['gender'].replace(0, test['gender'].mean())

# train = train.fillna(0)
# test = test.fillna(0)

# Delete date for now
train = train.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], axis=1)
test = test.drop(['transaction_date', 'membership_expire_date', 'registration_init_time'], axis=1)

# Create 4 new features
train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

train = train.drop(['payment_method_id2',
                    'payment_method_id3',
                    'payment_method_id4',
                    'payment_method_id5',
                    'payment_method_id6',
                    'payment_method_id8',
                    'payment_method_id10',
                    'payment_method_id11',
                    'payment_method_id12',
                    'payment_method_id13',
                    'payment_method_id14',
                    'payment_method_id16',
                    'payment_method_id17',
                    'payment_method_id18',
                    'payment_method_id19',
                    'payment_method_id20',
                    'payment_method_id21',
                    'payment_method_id22',
                    'payment_method_id23',
                    'payment_method_id24',
                    'payment_method_id25',
                    'payment_method_id27',
                    'payment_method_id28',
                    'payment_method_id31',
                    'payment_method_id33',
                    'payment_method_id34',
                    'transaction_date_day',
                    'membership_expire_date_day'], axis=1)

test = test.drop(['payment_method_id2',
                  'payment_method_id3',
                  'payment_method_id4',
                  'payment_method_id5',
                  'payment_method_id6',
                  'payment_method_id8',
                  'payment_method_id10',
                  'payment_method_id11',
                  'payment_method_id12',
                  'payment_method_id13',
                  'payment_method_id14',
                  'payment_method_id16',
                  'payment_method_id17',
                  'payment_method_id18',
                  'payment_method_id19',
                  'payment_method_id20',
                  'payment_method_id21',
                  'payment_method_id22',
                  'payment_method_id23',
                  'payment_method_id24',
                  'payment_method_id25',
                  'payment_method_id27',
                  'payment_method_id28',
                  'payment_method_id31',
                  'payment_method_id33',
                  'payment_method_id34',
                  'transaction_date_day',
                  'membership_expire_date_day'], axis=1)

feature_list = [
    # raw data
    'msno', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew',
    'is_cancel', 'city', 'bd', 'gender', 'registered_via', 'is_churn',
    # advanced features
    # user_log
    'log_day', 'total_25_sum', 'total_50_sum', 'total_75_sum', 'total_985_sum', 'total_100_sum', 'total_unq_sum',
    'total_secs_sum',
    'total_sum', 'total_25ratio', 'total_100ratio', 'persong_play', 'persong_time', 'daily_play', 'daily_listentime',
    'one_week_sum', 'two_week_sum', 'one_week_secs_sum', 'two_week_secs_sum', 'week_secs_sum_ratio', 'week_sum_ratio',
    'one_semimonth_sum', 'two_semimonth_sum', 'one_semimonth_secs_sum', 'two_semimonth_secs_sum',
    'semimonth_secs_sum_ratio', 'semimonth_sum_ratio',
    # transactions
    'discount', 'amt_per_day', 'is_discount', 'membership_days',
    'transaction_date_year', 'transaction_date_month', 'transaction_date_day',
    'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_day'
    # members
]

cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

params = {
    'base_score': 0.5,
    'eta': 0.002,
    'max_depth': 6,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bytree': 1.0,
    'gamma': 1,
    'max_child_weight': 5,
    'n_estimators': 600,
    'reg_alpha': '0',
    'reg_lambda': '1',
    'scale_pos_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 2017,
    'silent': True
}
x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3,
                                                          random_state=2017)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
cv_output = xgb.cv(params, xgb.DMatrix(x1, y1), num_boost_round=1500, early_stopping_rounds=20, verbose_eval=50,
                   show_stdv=False)
model = xgb.train(params, xgb.DMatrix(x1, y1), 2500, watchlist, feval=xgb_score, maximize=False, verbose_eval=50,
                  early_stopping_rounds=50)

pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

test['is_churn'] = pred.clip(0.0000001, 0.999999)
print(len(test))
test[['msno', 'is_churn']].to_csv('submission_xgboost_all_features_selection_eta_0.002_round_2500_Dec_15.csv',
                                  index=False)
