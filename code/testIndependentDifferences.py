# -*- coding: utf-8 -*-
"""

@author: 
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    TODO: package as function; set function so that it can do effect magnitude
    and n-samples ranges at the same time; settings for various inputs like effect,
    samples, number of simulations etc.
    
"""

# %% Import packages

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import pandas as pd
import seaborn as sns

# %% Sample way of independent tests and power with manipulated effect magnitude

"""

NOTE: second example below improves on this code by removing need for looping through simulations

"""

#For independent t-test on mean differences

# #Set mean and SD parameters of two groups
# #Assumes no treatment effect
# mu1 = 900
# sigma1 = 30
# mu2 = 900
# sigma2 = 30

# #Set the desired or 'true' effect size
# effectMagnitude = -10

#Set a range of desired effect magnitudes
effectMagnitude = np.array((range(-5,-40, -5)))

#Loop through effect magnitudes and extract statistical power
power = []
for effect in effectMagnitude:

    #Set mean and SD parameters of two groups
    #Assumes an effect specified above
    mu1 = 900
    sigma1 = 30
    mu2 = mu1 + effect
    sigma2 = 30
    
    #Set the null value
    nullVal = 0
    
    #Set the alpha level
    alpha = 0.05
    
    #Set the sample size for simulation
    n = 10
    
    #Set the number of simulations to run
    nSims = 100000
    
    #Run through simulations to extract mean differences
    meanDiff = np.zeros(nSims)
    stdErrEst = np.zeros(nSims)
    for simNo in range(nSims):
    
        #Randomly sample from normal distributions with mean and SD parameters
        np.random.seed(111 * (simNo+1))
        samples1 = np.random.normal(mu1, sigma1, n)
        np.random.seed(222 * (simNo+1))
        samples2 = np.random.normal(mu2, sigma2, n)
        
        #Calculate mean difference
        meanDiff[simNo] = samples2.mean() - samples1.mean()
        
        #Calculate estimated standard error from pooled SD
        sdPooled = np.sqrt((samples1.std()**2 + samples2.std()**2) / 2)
        stdErrEst[simNo] = np.sqrt((sdPooled**2/n)+(sdPooled**2/n))
        
    #Calculate t-statistic
    tStat = (meanDiff - nullVal) / stdErrEst
    
    #Calculate two-sided p-value
    pVal = stats.t.sf(np.abs(tStat), n-1)
    
    # #Create histogram of mean differences
    # plt.hist(meanDiff, bins = 50)
    
    # #Create histogram of t-statistic
    # plt.hist(tStat, bins = 50)
    
    # #Create histogram of p-value
    # plt.hist(pVal, bins = 50)
    
    #Calculate proportion of p-values < alpha
    #This equates to statistical power
    power.append(np.sum(pVal < alpha) / nSims)
    
#Plot effect magnitude against power
#Reversed x-axis isn't actually necessary here
plt.plot(effectMagnitude, np.array(power),
         c = 'red', ls = '-', lw = 2, marker = 'o', markersize = 10)
plt.gca().invert_xaxis()

# %% Sample way of independent tests and power with manipulated sample size

#For independent t-test on mean differences

#Set the desired or 'true' effect size
effectMagnitude = -10

#Set sample sizes to work through
nSamples = np.array(range(10,100,5))

#Loop through effect magnitudes and extract statistical power
power = []
for n in nSamples:

    #Set mean and SD parameters of two groups
    #Assumes an effect specified above
    mu1 = 900
    sigma1 = 30
    mu2 = mu1 + effectMagnitude
    sigma2 = 30
    
    #Set the null value
    nullVal = 0
    
    #Set the alpha level
    alpha = 0.05
    
    #Set the number of simulations to run
    nSims = 100000
    
    #Run through simulations to extract mean differences
    # meanDiff = np.zeros(nSims)
    # stdErrEst = np.zeros(nSims)

    #Randomly sample from normal distributions with mean and SD parameters
    np.random.seed(111 * (simNo+1))
    samples1 = np.random.normal(mu1, sigma1, (n, nSims))
    np.random.seed(222 * (simNo+1))
    samples2 = np.random.normal(mu2, sigma2, (n, nSims))
        
    #Calculate mean difference
    meanDiff = samples2.mean(axis = 0) - samples1.mean(axis = 0)

    #Calculate estimated standard error from pooled SD
    sdPooled = np.sqrt((samples1.std(axis = 0)**2 + samples2.std(axis = 0)**2) / 2)
    stdErrEst = np.sqrt((sdPooled**2/n)+(sdPooled**2/n))
        
    #Calculate t-statistic
    tStat = (meanDiff - nullVal) / stdErrEst
    
    #Calculate two-sided p-value
    pVal = stats.t.sf(np.abs(tStat), n-1)
    
    # #Create histogram of mean differences
    # plt.hist(meanDiff, bins = 50)
    
    # #Create histogram of t-statistic
    # plt.hist(tStat, bins = 50)
    
    # #Create histogram of p-value
    # plt.hist(pVal, bins = 50)
    
    #Calculate proportion of p-values < alpha
    #This equates to statistical power
    power.append(np.sum(pVal < alpha) / nSims)
    
#Plot effect magnitude against power
#Reversed x-axis isn't actually necessary here
plt.plot(nSamples, np.array(power),
         c = 'red', ls = '-', lw = 2, marker = 'o', markersize = 10)

# %% Sample way of independent tests and power with manipulated effect and sample size

#For independent t-test on mean differences

#Set the desired or 'true' effect size as list
effectMagnitude = [-10, -20, -30, -40]

#Set sample sizes to work through as list
nSamples = [10, 20, 30, 40]

#Create combination of effect and n
nPlusEffect = list(itertools.product(nSamples,effectMagnitude))

#Create dictionary to store values in
powerDict = {'n': [], 'effect': [], 'power': []}

#Loop through sample size and effect magnitude combinations to extract statistical power
for testSample in nPlusEffect:
    
    #Extract iterative n and effect
    n = testSample[0]
    effect = testSample[1]

    #Set mean and SD parameters of two groups
    #Assumes an effect specified above
    mu1 = 900
    sigma1 = 30
    mu2 = mu1 + effect
    sigma2 = 30
    
    #Set the null value
    nullVal = 0
    
    #Set the alpha level
    alpha = 0.05
    
    #Set the number of simulations to run
    nSims = 100000

    #Randomly sample from normal distributions with mean and SD parameters
    np.random.seed(111 * (simNo+1))
    samples1 = np.random.normal(mu1, sigma1, (n, nSims))
    np.random.seed(222 * (simNo+1))
    samples2 = np.random.normal(mu2, sigma2, (n, nSims))
        
    #Calculate mean difference
    meanDiff = samples2.mean(axis = 0) - samples1.mean(axis = 0)

    #Calculate estimated standard error from pooled SD
    sdPooled = np.sqrt((samples1.std(axis = 0)**2 + samples2.std(axis = 0)**2) / 2)
    stdErrEst = np.sqrt((sdPooled**2/n)+(sdPooled**2/n))
        
    #Calculate t-statistic
    tStat = (meanDiff - nullVal) / stdErrEst
    
    #Calculate two-sided p-value
    pVal = stats.t.sf(np.abs(tStat), n-1)
    
    # #Create histogram of mean differences
    # plt.hist(meanDiff, bins = 50)
    
    # #Create histogram of t-statistic
    # plt.hist(tStat, bins = 50)
    
    # #Create histogram of p-value
    # plt.hist(pVal, bins = 50)
    
    #Calculate proportion of p-values < alpha
    #This equates to statistical power
    power = np.sum(pVal < alpha) / nSims
    
    #Append data to dictionary
    powerDict['n'].append(n)
    powerDict['effect'].append(effect)
    powerDict['power'].append(power)
    
#Convert to dataframe
powerDf = pd.DataFrame.from_dict(powerDict)
    
#Plot power as a function of sample size and effect
#### Not pretty but it gets the point for now
sns.lineplot(data = powerDf, x = 'n', y = 'power', hue = 'effect')



# %%% ----- End of testIndependentDifferences.py -----