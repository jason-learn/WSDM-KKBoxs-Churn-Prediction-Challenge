import numpy as np
import pandas as pd


def process_user_log_together(df):
    """
    After union all chunk file, do sum again.
    :param df:
    :return:
    """

    df = df.fillna(0)

    grouped_object = df.groupby('msno', sort=False)  # not sorting results in a minor speedup
    func = {'log_day_monthly': ['sum'],
            'total_25_sum_monthly': ['sum'],
            'total_50_sum_monthly': ['sum'],
            'total_75_sum_monthly': ['sum'],
            'total_985_sum_monthly': ['sum'],
            'total_100_sum_monthly': ['sum'],
            'total_unq_sum_monthly': ['sum'],
            'total_secs_sum_monthly': ['sum']
            }
    user_log_all = grouped_object.agg(func).reset_index()
    user_log_all.columns = ['_'.join(col).strip() for col in user_log_all.columns.values]
    user_log_all.rename(columns={'msno_': 'msno',
                                 'log_day_monthly_sum': 'log_day_monthly',
                                 'total_25_sum_monthly_sum': 'total_25_sum_monthly',
                                 'total_50_sum_monthly_sum': 'total_50_sum_monthly',
                                 'total_75_sum_monthly_sum': 'total_75_sum_monthly',
                                 'total_985_sum_monthly_sum': 'total_985_sum_monthly',
                                 'total_100_sum_monthly_sum': 'total_100_sum_monthly',
                                 'total_unq_sum_monthly_sum': 'total_unq_sum_monthly',
                                 'total_secs_sum_monthly_sum': 'total_secs_sum_monthly',
                                 }, inplace=True)

    return user_log_all


def calculate_user_log_features(train):
    """
    Calculate the user log features.
    :param train:
    :return:
    """
    train['total_monthly_sum'] = train['total_25_sum_monthly'] + train['total_50_sum_monthly'] + train[
        'total_75_sum_monthly'] + train['total_985_sum_monthly'] + train['total_100_sum_monthly']

    # Monthly Habit for listening to music
    train['total_25_ratio'] = train['total_25_sum_monthly'] / train['total_monthly_sum']
    train['total_100_ratio'] = train['total_100_sum_monthly'] / train['total_monthly_sum']

    # 听歌是循环播放还是试听,每首歌播放次数
    train['persong_play'] = train['total_monthly_sum'] / train['total_unq_sum_monthly']

    # 听歌每首歌平均播放时间
    train['persong_time'] = train['total_secs_sum_monthly'] / train['total_monthly_sum']

    # 平均每天听歌数量
    train['daily_play'] = train['total_monthly_sum'] / train['log_day_monthly']

    # 平均每天听歌时间
    train['daily_listentime'] = train['total_secs_sum_monthly'] / train['log_day_monthly']

    train.replace(np.inf, 0, inplace=True)
    train = train.fillna(0)

    return train


train = pd.read_csv('../input/processed_user_log_mid_all.csv')
user_log_test = pd.read_csv('../input/processed_user_log_mid_all.csv')
user_log_test = user_log_test[['msno',
                               'log_day_monthly',
                               'total_25_sum_monthly',
                               'total_50_sum_monthly',
                               'total_75_sum_monthly',
                               'total_985_sum_monthly',
                               'total_100_sum_monthly',
                               'total_unq_sum_monthly',
                               'total_secs_sum_monthly']]

print(train.columns)
print(user_log_test.columns)

train = train.append(user_log_test)

train = process_user_log_together(train)

train = calculate_user_log_features(train)

print(len(train))

train.to_csv('../input/processed_features_user_log_all_time_including_mar.csv', index=False)
