import pandas as pd

'''
# LB 0.12432 CV 0.122651 Train LogLoss 0.103781
file1 = pd.read_csv('result/submission_lightgbm_all_time_feaetures_origin_version_eta_0.002_round_2500_Dec_16.csv')
weight1 = 0.30

# LB 0.12383 CV 0.127227
file2 = pd.read_csv('result/submission_lightgbm_features_trans_user_log_split_by_month_eta_0.002_round_2500_Dec_15.csv')
weight2 = 0.30

# LB 0.12323 Train LogLoss 0.0966805
file3 = pd.read_csv('result/submission_lightgbm_features_all_eta_0.002_round_2000_Dec_13.csv')
weight3 = 0.2

# LB 0.12705 CV 0.136615 Train LogLoss 0.094903
file4 = pd.read_csv('result/submission_xgboost_user_log_transaction_features_eta_0.002_round_2500_Dec_11.csv')
weight4 = 0.2

file1['is_churn'] = file1['is_churn'] * weight1 + file2['is_churn'] * weight2 + \
                    file3['is_churn'] * weight3 + file4['is_churn'] * weight4

file1.to_csv('submission_weight_avg_4_0.3_0.3_0.2_0.2.csv', index=False)
'''

# LB 0.12432 CV 0.122651 Train LogLoss 0.103781
file1 = pd.read_csv('result/submission_lightgbm_all_time_feaetures_origin_version_eta_0.002_round_2500_Dec_16.csv')
weight1 = 0.28

# LB 0.12383 CV 0.127227
file2 = pd.read_csv('result/submission_lightgbm_features_trans_user_log_split_by_month_eta_0.002_round_2500_Dec_15.csv')
weight2 = 0.28

# LB 0.12393 CV 0.122639 Train LogLoss 0.102916
file3 = pd.read_csv('result/submission_lightgbm_features_selection_origin_version_eta_0.002_round_2500_Dec_17.csv')
weight3 = 0.44

file1['is_churn'] = file1['is_churn'] * weight1 + file2['is_churn'] * weight2 + \
                    file3['is_churn'] * weight3

file1.to_csv('submission_weight_avg_0.44_0.28_0.28.csv', index=False)
