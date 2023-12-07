import matplotlib.pyplot as plt

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
    
def plot_remote_jobs_by_state(jobs_df, treshold=None):
    counts = jobs_df.groupby(['st_code', 'is_remote']).size().unstack()
    counts['total'] = counts.sum(axis=1)

    if treshold is not None and isinstance(treshold, int):
        filtered_counts = counts[counts['total'] < treshold].copy()
    else:
        filtered_counts = counts.copy()

    filtered_counts.sort_values(by='total', ascending=False, inplace=True)

    del filtered_counts['total']
    filtered_counts.plot(kind='bar', stacked=True, figsize=(15, 10))

    plt.xlabel('State Code')
    plt.ylabel('Number of Jobs')
    plt.title('Number of Remote and Non-Remote Jobs by State')

    plt.legend(['Non-Remote', 'Remote'], loc='upper right')
    plt.show()