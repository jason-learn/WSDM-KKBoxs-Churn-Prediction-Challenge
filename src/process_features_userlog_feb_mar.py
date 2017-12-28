import gc

import numpy as np
import pandas as pd


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

    train['one_week_sum'] = train['one_week_total_25_sum'] + train['one_week_total_50_sum'] + train[
        'one_week_total_75_sum'] + train['one_week_total_985_sum'] + train['one_week_total_100_sum']

    train['two_week_sum'] = train['two_week_total_25_sum'] + train['two_week_total_50_sum'] + train[
        'two_week_total_75_sum'] + train['two_week_total_985_sum'] + train['two_week_total_100_sum']

    # 第四周听歌时间与第三周比较
    train['week_secs_sum_ratio'] = train['two_week_total_secs_sum'] / train['one_week_total_secs_sum']
    # 第四周听歌数与第三周比较
    train['week_sum_ratio'] = train['two_week_sum'] / train['one_week_sum']

    train['one_semimonth_sum'] = train['one_semimonth_total_25_sum'] + train['one_semimonth_total_50_sum'] \
                                 + train['one_semimonth_total_75_sum'] + train[
                                     'one_semimonth_total_985_sum'] + train['one_semimonth_total_100_sum']

    train['two_semimonth_sum'] = train['two_semimonth_total_25_sum'] + train['two_semimonth_total_50_sum'] \
                                 + train['two_semimonth_total_75_sum'] + train[
                                     'two_semimonth_total_985_sum'] + train['two_semimonth_total_100_sum']

    # 第二个半月听歌时间与第一个半月比较
    train['semimonth_secs_sum_ratio'] = train['two_semimonth_total_secs_sum'] / train['one_semimonth_total_secs_sum']
    # 第二个半月听歌数与第一个半月比较
    train['semimonth_sum_ratio'] = train['two_semimonth_sum'] / train['one_semimonth_sum']

    train.replace(np.inf, 0, inplace=True)
    train = train.fillna(0)
    train = train.drop(['log_day_monthly',
                        'total_25_sum_monthly',
                        'total_50_sum_monthly',
                        'total_75_sum_monthly',
                        'total_985_sum_monthly',
                        'total_100_sum_monthly',
                        'total_unq_sum_monthly',
                        'total_secs_sum_monthly',
                        'one_week_log_day',
                        'one_week_total_25_sum',
                        'one_week_total_50_sum',
                        'one_week_total_75_sum',
                        'one_week_total_985_sum',
                        'one_week_total_100_sum',
                        'one_week_total_unq_sum',
                        'one_week_total_secs_sum',
                        'two_week_log_day',
                        'two_week_total_25_sum',
                        'two_week_total_50_sum',
                        'two_week_total_75_sum',
                        'two_week_total_985_sum',
                        'two_week_total_100_sum',
                        'two_week_total_unq_sum',
                        'two_week_total_secs_sum',
                        'one_semimonth_log_day',
                        'one_semimonth_total_25_sum',
                        'one_semimonth_total_50_sum',
                        'one_semimonth_total_75_sum',
                        'one_semimonth_total_985_sum',
                        'one_semimonth_total_100_sum',
                        'one_semimonth_total_unq_sum',
                        'one_semimonth_total_secs_sum',
                        'two_semimonth_log_day',
                        'two_semimonth_total_25_sum',
                        'two_semimonth_total_50_sum',
                        'two_semimonth_total_75_sum',
                        'two_semimonth_total_985_sum',
                        'two_semimonth_total_100_sum',
                        'two_semimonth_total_unq_sum',
                        'two_semimonth_total_secs_sum'], axis=1)

    return train


train = pd.read_csv('../input/processed_user_log_feb.csv')

train = calculate_user_log_features(train)

train.to_csv('../input/processed_features_user_log_feb.csv', index=False)

del train
gc.collect()

test = pd.read_csv('../input/processed_user_log_mar.csv')

test = calculate_user_log_features(test)

test.to_csv('../input/processed_features_user_log_mar.csv', index=False)
