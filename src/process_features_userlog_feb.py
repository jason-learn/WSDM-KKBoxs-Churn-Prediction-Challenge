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

    # Stage 2: Week Total Data
    # Divided DataFrame by Two Week
    one_week = df[(df['date'] < 20170220) & (df['date'] > 20170212)]

    grouped_object = one_week.groupby('msno', sort=False)
    one_week = grouped_object.agg(func).reset_index()
    one_week.columns = ['_'.join(col).strip() for col in one_week.columns.values]
    one_week.rename(columns={'msno_': 'msno',
                             'date_count': 'one_week_log_day',
                             'num_25_sum': 'one_week_total_25_sum',
                             'num_50_sum': 'one_week_total_50_sum',
                             'num_75_sum': 'one_week_total_75_sum',
                             'num_985_sum': 'one_week_total_985_sum',
                             'num_100_sum': 'one_week_total_100_sum',
                             'num_unq_sum': 'one_week_total_unq_sum',
                             'total_secs_sum': 'one_week_total_secs_sum'}, inplace=True)

    one_month = pd.merge(one_month, one_week, on=['msno'], how='left')

    del one_week
    gc.collect()

    two_week = df[(df['date'] < 20170227) & (df['date'] > 20170219)]

    grouped_object = two_week.groupby('msno', sort=False)
    two_week = grouped_object.agg(func).reset_index()
    two_week.columns = ['_'.join(col).strip() for col in two_week.columns.values]
    two_week.rename(columns={'msno_': 'msno',
                             'date_count': 'two_week_log_day',
                             'num_25_sum': 'two_week_total_25_sum',
                             'num_50_sum': 'two_week_total_50_sum',
                             'num_75_sum': 'two_week_total_75_sum',
                             'num_985_sum': 'two_week_total_985_sum',
                             'num_100_sum': 'two_week_total_100_sum',
                             'num_unq_sum': 'two_week_total_unq_sum',
                             'total_secs_sum': 'two_week_total_secs_sum'}, inplace=True)

    one_month = pd.merge(one_month, two_week, on=['msno'], how='left')

    del two_week
    gc.collect()

    # Stage 3: Semimonth Total Data
    one_semimonth = df[(df['date'] < 20170215) & (df['date'] > 20170131)]

    grouped_object = one_semimonth.groupby('msno', sort=False)
    one_semimonth = grouped_object.agg(func).reset_index()
    one_semimonth.columns = ['_'.join(col).strip() for col in one_semimonth.columns.values]
    one_semimonth.rename(columns={'msno_': 'msno',
                                  'date_count': 'one_semimonth_log_day',
                                  'num_25_sum': 'one_semimonth_total_25_sum',
                                  'num_50_sum': 'one_semimonth_total_50_sum',
                                  'num_75_sum': 'one_semimonth_total_75_sum',
                                  'num_985_sum': 'one_semimonth_total_985_sum',
                                  'num_100_sum': 'one_semimonth_total_100_sum',
                                  'num_unq_sum': 'one_semimonth_total_unq_sum',
                                  'total_secs_sum': 'one_semimonth_total_secs_sum'}, inplace=True)

    one_month = pd.merge(one_month, one_semimonth, on=['msno'], how='left')

    del one_semimonth
    gc.collect()

    two_semimonth = df[(df['date'] < 20170301) & (df['date'] > 20170214)]

    grouped_object = two_semimonth.groupby('msno', sort=False)
    two_semimonth = grouped_object.agg(func).reset_index()
    two_semimonth.columns = ['_'.join(col).strip() for col in two_semimonth.columns.values]
    two_semimonth.rename(columns={'msno_': 'msno',
                                  'date_count': 'two_semimonth_log_day',
                                  'num_25_sum': 'two_semimonth_total_25_sum',
                                  'num_50_sum': 'two_semimonth_total_50_sum',
                                  'num_75_sum': 'two_semimonth_total_75_sum',
                                  'num_985_sum': 'two_semimonth_total_985_sum',
                                  'num_100_sum': 'two_semimonth_total_100_sum',
                                  'num_unq_sum': 'two_semimonth_total_unq_sum',
                                  'total_secs_sum': 'two_semimonth_total_secs_sum'}, inplace=True)

    one_month = pd.merge(one_month, two_semimonth, on=['msno'], how='left')

    del two_semimonth
    gc.collect()

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
            'total_secs_sum_monthly': ['sum'],
            'one_week_log_day': ['sum'],
            'one_week_total_25_sum': ['sum'],
            'one_week_total_50_sum': ['sum'],
            'one_week_total_75_sum': ['sum'],
            'one_week_total_985_sum': ['sum'],
            'one_week_total_100_sum': ['sum'],
            'one_week_total_unq_sum': ['sum'],
            'one_week_total_secs_sum': ['sum'],
            'two_week_log_day': ['sum'],
            'two_week_total_25_sum': ['sum'],
            'two_week_total_50_sum': ['sum'],
            'two_week_total_75_sum': ['sum'],
            'two_week_total_985_sum': ['sum'],
            'two_week_total_100_sum': ['sum'],
            'two_week_total_unq_sum': ['sum'],
            'two_week_total_secs_sum': ['sum'],
            'one_semimonth_log_day': ['sum'],
            'one_semimonth_total_25_sum': ['sum'],
            'one_semimonth_total_50_sum': ['sum'],
            'one_semimonth_total_75_sum': ['sum'],
            'one_semimonth_total_985_sum': ['sum'],
            'one_semimonth_total_100_sum': ['sum'],
            'one_semimonth_total_unq_sum': ['sum'],
            'one_semimonth_total_secs_sum': ['sum'],
            'two_semimonth_log_day': ['sum'],
            'two_semimonth_total_25_sum': ['sum'],
            'two_semimonth_total_50_sum': ['sum'],
            'two_semimonth_total_75_sum': ['sum'],
            'two_semimonth_total_985_sum': ['sum'],
            'two_semimonth_total_100_sum': ['sum'],
            'two_semimonth_total_unq_sum': ['sum'],
            'two_semimonth_total_secs_sum': ['sum']
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
                                 'one_week_log_day_sum': 'one_week_log_day',
                                 'one_week_total_25_sum_sum': 'one_week_total_25_sum',
                                 'one_week_total_50_sum_sum': 'one_week_total_50_sum',
                                 'one_week_total_75_sum_sum': 'one_week_total_75_sum',
                                 'one_week_total_985_sum_sum': 'one_week_total_985_sum',
                                 'one_week_total_100_sum_sum': 'one_week_total_100_sum',
                                 'one_week_total_unq_sum_sum': 'one_week_total_unq_sum',
                                 'one_week_total_secs_sum_sum': 'one_week_total_secs_sum',
                                 'two_week_log_day_sum': 'two_week_log_day',
                                 'two_week_total_25_sum_sum': 'two_week_total_25_sum',
                                 'two_week_total_50_sum_sum': 'two_week_total_50_sum',
                                 'two_week_total_75_sum_sum': 'two_week_total_75_sum',
                                 'two_week_total_985_sum_sum': 'two_week_total_985_sum',
                                 'two_week_total_100_sum_sum': 'two_week_total_100_sum',
                                 'two_week_total_unq_sum_sum': 'two_week_total_unq_sum',
                                 'two_week_total_secs_sum_sum': 'two_week_total_secs_sum',
                                 'one_semimonth_log_day_sum': 'one_semimonth_log_day',
                                 'one_semimonth_total_25_sum_sum': 'one_semimonth_total_25_sum',
                                 'one_semimonth_total_50_sum_sum': 'one_semimonth_total_50_sum',
                                 'one_semimonth_total_75_sum_sum': 'one_semimonth_total_75_sum',
                                 'one_semimonth_total_985_sum_sum': 'one_semimonth_total_985_sum',
                                 'one_semimonth_total_100_sum_sum': 'one_semimonth_total_100_sum',
                                 'one_semimonth_total_unq_sum_sum': 'one_semimonth_total_unq_sum',
                                 'one_semimonth_total_secs_sum_sum': 'one_semimonth_total_secs_sum',
                                 'two_semimonth_log_day_sum': 'two_semimonth_log_day',
                                 'two_semimonth_total_25_sum_sum': 'two_semimonth_total_25_sum',
                                 'two_semimonth_total_50_sum_sum': 'two_semimonth_total_50_sum',
                                 'two_semimonth_total_75_sum_sum': 'two_semimonth_total_75_sum',
                                 'two_semimonth_total_985_sum_sum': 'two_semimonth_total_985_sum',
                                 'two_semimonth_total_100_sum_sum': 'two_semimonth_total_100_sum',
                                 'two_semimonth_total_unq_sum_sum': 'two_semimonth_total_unq_sum',
                                 'two_semimonth_total_secs_sum_sum': 'two_semimonth_total_secs_sum'
                                 }, inplace=True)

    return user_log_all


gc.enable()

size = 1e6
reader = pd.read_csv('../input/user_log_feb.csv', chunksize=size)
start_time = time.time()
for i in range(17):  # 17
    user_log_chunk = next(reader)
    if i == 0:
        user_log_feb = process_user_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        user_log_feb = user_log_feb.append(process_user_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del user_log_chunk

user_log_feb = process_user_log_together(user_log_feb)

user_log_feb.to_csv("../input/processed_user_log_feb.csv", index=False)

print('Done')
