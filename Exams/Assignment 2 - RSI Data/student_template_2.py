"""
CULMINATING ASSIGNMENT 2: RSI

Create a program which will provide answers to the tasks posed in the assignment description.

For this assignment, you will start from scratch with this blank template.
It is your reponsibility to include all necessary program components in this file.
Your code needs to be successful as well as sufficiently commented/documented to receive full credit.

The relevant data file is named "all_participant_data_rsi.csv" and is located in data > drop-jump.

All file paths need to be relative and execute from this file's directory;
the Visual Studio Code setting "Python > Terminal: Execute in File Dir"
must be enabled in File > Preferences > Settings.

See Canvas for additional requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
from scipy.stats import chisquare, norm, sem, t, ttest_1samp


# Check for the data file relative to the assignment folder, then the repository root.
data_path = '../../data/drop-jump/all_participant_data_rsi.csv'
if not exists(data_path):
    data_path = 'data/drop-jump/all_participant_data_rsi.csv'

df = pd.read_csv(data_path)
force_plate_rsi = df['force_plate_rsi']
accelerometer_rsi = df['accelerometer_rsi']
alpha = 0.05


# Task 1: paired t-test using the differences between measurements.
# The null hypothesis is that the mean paired difference is zero.
# The alternative hypothesis is that the mean paired difference is not zero.
difference = force_plate_rsi - accelerometer_rsi
t_value, p_value = ttest_1samp(difference, popmean=0, alternative='two-sided')

print('Task 1: Paired t-test')
print('Null hypothesis: The mean difference between the methods is 0.')
print('Alternative hypothesis: The mean difference between the methods is not 0.')
print('t-value:', t_value)
print('p-value:', p_value)
if p_value < alpha:
    print('Reject the null hypothesis.')
else:
    print('Do not reject the null hypothesis.')


# Task 2: fit a normal distribution and compare observed and expected bin counts.
accelerometer_values = accelerometer_rsi.to_numpy()
fitted_mean, fitted_std = norm.fit(accelerometer_values)
bins = np.r_[-np.inf,
             np.linspace(fitted_mean - 2 * fitted_std,
                         fitted_mean + 2 * fitted_std, 9),
             np.inf]
observed_counts, _ = np.histogram(accelerometer_values, bins=bins)
expected_probabilities = np.diff(norm.cdf(bins, loc=fitted_mean, scale=fitted_std))
expected_counts = expected_probabilities * len(accelerometer_values)
# ddof=2 accounts for estimating the mean and standard deviation from the data.
chi2_stat, chi2_p_value = chisquare(
    f_obs=observed_counts, f_exp=expected_counts, ddof=2)

print('\nTask 2: Normal distribution chi-square goodness of fit')
print('Fitted mean:', fitted_mean)
print('Fitted standard deviation:', fitted_std)
print('Chi2 stat:', chi2_stat)
print('p-value:', chi2_p_value)
if chi2_p_value < alpha:
    print('The normal distribution line is not a good fit.')
else:
    print('The normal distribution line is a reasonable fit.')

x = np.linspace(min(accelerometer_values), max(accelerometer_values), 500)
plt.hist(accelerometer_values, bins=10, density=True,
         label='Accelerometer RSI')
plt.plot(x, norm.pdf(x, loc=fitted_mean, scale=fitted_std),
         label='Fitted normal distribution')
plt.xlabel('Accelerometer RSI')
plt.ylabel('Probability density')
plt.title('Accelerometer RSI and fitted normal distribution')
plt.legend()


# Task 3: select and print Participant F's trials.
# The OR conditions select each of Participant F's trials from the dataframe.
participant_f = df[(df['trial'] == 'F_trial_1') | (df['trial'] == 'F_trial_2') |
                   (df['trial'] == 'F_trial_3') | (df['trial'] == 'F_trial_4') |
                   (df['trial'] == 'F_trial_5') | (df['trial'] == 'F_trial_6') |
                   (df['trial'] == 'F_trial_7') | (df['trial'] == 'F_trial_8') |
                   (df['trial'] == 'F_trial_9') | (df['trial'] == 'F_trial_10')]
print('\nTask 3: Participant F trials')
print(participant_f)

force_f = participant_f['force_plate_rsi']
accelerometer_f = participant_f['accelerometer_rsi']
force_mean = force_f.mean()
accelerometer_mean = accelerometer_f.mean()
# Use the t-distribution and standard error to calculate a 95% interval for each mean.
force_ci = t.interval(confidence=0.95, df=len(force_f) - 1,
                      loc=force_mean, scale=sem(force_f))
accelerometer_ci = t.interval(confidence=0.95, df=len(accelerometer_f) - 1,
                              loc=accelerometer_mean, scale=sem(accelerometer_f))
force_lower_ci, force_upper_ci = force_ci
accelerometer_lower_ci, accelerometer_upper_ci = accelerometer_ci

print('Participant F force plate mean RSI 95% confidence interval:',
      force_lower_ci, 'to', force_upper_ci)
print('Participant F accelerometer mean RSI 95% confidence interval:',
      accelerometer_lower_ci, 'to', accelerometer_upper_ci)

plt.show()
