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
