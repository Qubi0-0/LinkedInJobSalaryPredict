import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Converts hourly, monthly, and weekly salaries to yearly salaries
def convert_to_yearly(sal, pay_period):
    if pay_period == 'HOURLY':
        return sal * 2080
    elif pay_period == 'MONTHLY':
        return sal * 12
    elif pay_period == 'WEEKLY':
        return sal * 52
    else:
        return sal
    
# Encode job titles
def encode_job_ttls(jobs_df, print_stats=False):
    mean_salary = jobs_df.groupby('xp_lvl')['med_sal'].mean()
    mean_salary_sorted = mean_salary.sort_values(ascending=False)
    if print_stats:
        most_significant_job = mean_salary_sorted.index[0]
        print('Mean salary by experience level:')
        print(mean_salary_sorted)
        print(most_significant_job)
        print('\n')
    jobs_df = jobs_df.sort_values(by='med_sal')

    le = LabelEncoder()
    jobs_df['Job_Ttl'] = le.fit_transform(jobs_df['Job_Ttl'])
    
    return jobs_df

# Normalize data
def normalize_data(jobs_df):
    scaler = MinMaxScaler()
    jobs_df['med_sal'] = scaler.fit_transform(jobs_df['med_sal'].values.reshape(-1, 1))
    jobs_df['exp_lvl'] = scaler.fit_transform(jobs_df['exp_lvl'].values.reshape(-1, 1))
    return jobs_df

