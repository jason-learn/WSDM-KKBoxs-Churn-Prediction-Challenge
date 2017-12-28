import gc
import time

import pandas as pd


def process_user_log(df):
    """
    Only do simple sum. mean operation.
    :param df: chunk dataframe from very large file.
    :return: processed dataframe
    """

    # Divided DataFrame by date
    # train = train[(train['date'] < 20170301) & (train['date'] > 20170131)]

    # Stage 1: One Month Total Data
    grouped_object = df.groupby('msno', sort=False)  # not sorting results in a minor speedup
    func = {'date': ['count'],
            'num_25': ['sum'], 'num_50': ['sum'],
            'num_75': ['sum'], 'num_985': ['sum'],
            'num_100': ['sum'], 'num_unq': ['sum'], 'total_secs': ['sum']}
    one_month = grouped_object.agg(func).reset_index()
    one_month.columns = ['_'.join(col).strip() for col in one_month.columns.values]
    one_month.rename(columns={'msno_': 'msno',
                              'date_count': 'log_day_monthly',
                              'num_25_sum': 'total_25_sum_monthly',
                              'num_50_sum': 'total_50_sum_monthly',
                              'num_75_sum': 'total_75_sum_monthly',
                              'num_985_sum': 'total_985_sum_monthly',
                              'num_100_sum': 'total_100_sum_monthly',
                              'num_unq_sum': 'total_unq_sum_monthly',
                              'total_secs_sum': 'total_secs_sum_monthly'}, inplace=True)

    return one_month


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


gc.enable()

size = 4e7  # 40 million
reader = pd.read_csv('../input/user_logs.csv', chunksize=size)
start_time = time.time()
for i in range(10):
    user_log_chunk = next(reader)
    if i == 0:
        user_log_feb = process_user_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        user_log_feb = user_log_feb.append(process_user_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

user_log_feb = process_user_log_together(user_log_feb)

print(len(user_log_feb))

user_log_feb.to_csv("../input/processed_user_log_mid_all.csv", index=False)

print('Done')
