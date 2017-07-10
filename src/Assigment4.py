#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:31:40 2017

@author: sophieyang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

'''
ticket_id - unique identifier for tickets
agency_name - Agency that issued the ticket
inspector_name - Name of inspector that issued the ticket
violator_name - Name of the person/organization that the ticket was issued to
violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
ticket_issued_date - Date and time the ticket was issued
hearing_date - Date and time the violator's hearing was scheduled
violation_code, violation_description - Type of violation
disposition - Judgment and judgement type
fine_amount - Violation fine amount, excluding fees
admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
late_fee - 10% fee assigned to responsible judgments
discount_amount - discount applied, if any
clean_up_cost - DPW clean-up or graffiti removal cost
judgment_amount - Sum of all fines and fees

'''

train = pd.read_csv('downloads/train.csv', encoding = 'ISO-8859-1')
test = pd.read_csv('downloads/test.csv')
addresses = pd.read_csv('downloads/addresses.csv')
latlons = pd.read_csv('downloads/latlons.csv')
    
# Clean up training data
train = train.drop(['payment_amount', 'payment_date', 'payment_status', 
                    'balance_due', 'collection_status', 'compliance_detail'], axis=1)
trainOrig = train
train = train[~np.isnan(train['compliance'])]

# Reset Index
train = train.reset_index()

# Get the number of NaN per column
pd.isnull(train.iloc[:,:]).sum()
pd.isnull(test.iloc[:,:]).sum()

# Remove columns with too many NaNs
train = train.drop(['violation_zip_code', 'non_us_str_code', 'grafitti_status'], axis=1)
test = test.drop(['violation_zip_code', 'non_us_str_code', 'grafitti_status'], axis=1)
    
# Remove other columns
train = train.drop('ticket_id', axis=1)
    
pd.isnull(train.iloc[:,:]).sum()
pd.isnull(test.iloc[:,:]).sum()
    
# Deal with State, City, Zipcode, Country
train['country'].unique() # ['USA', 'Cana', 'Aust', 'Egyp', 'Germ']
train['state'].unique()

# Plot some stuff
plt.figure();plt.hist(train['compliance']); plt.title('compliance')
plt.figure();plt.hist(train['fine_amount']);plt.title('fine_amount')
plt.figure();plt.hist(train['admin_fee']);plt.title('admin_fee')
plt.figure();plt.hist(train['state_fee']);plt.title('state_fee')
plt.figure();plt.hist(train['late_fee']);plt.title('late_fee')
plt.figure();plt.hist(train['discount_amount']);plt.title('discount_amount')
plt.figure();plt.hist(train['clean_up_cost']);plt.title('clean_up_cost')
plt.figure();plt.hist(train['judgment_amount']);plt.title('judgment_amount')

# Getting the cases that are compliant
trainCompliant = train[train['compliance']==1]
plt.figure();plt.hist(trainCompliant['fine_amount']);plt.title('Compliant fine_amount')
plt.figure();plt.hist(trainCompliant['admin_fee']);plt.title('Compliant admin_fee')
plt.figure();plt.hist(trainCompliant['state_fee']);plt.title('Compliant state_fee')
plt.figure();plt.hist(trainCompliant['late_fee']);plt.title('Compliant late_fee')
plt.figure();plt.hist(trainCompliant['discount_amount']);plt.title('Compliant discount_amount')
plt.figure();plt.hist(trainCompliant['clean_up_cost']);plt.title('Compliant clean_up_cost')
plt.figure();plt.hist(trainCompliant['judgment_amount']);plt.title('Compliant judgment_amount')

# Getting the cases that are compliant
trainNonCompliant = train[train['compliance']==0]
plt.figure();plt.hist(trainNonCompliant['fine_amount']);plt.title('Compliant fine_amount')
plt.figure();plt.hist(trainNonCompliant['admin_fee']);plt.title('Compliant admin_fee')
plt.figure();plt.hist(trainNonCompliant['state_fee']);plt.title('Compliant state_fee')
plt.figure();plt.hist(trainNonCompliant['late_fee']);plt.title('Compliant late_fee')
plt.figure();plt.hist(trainNonCompliant['discount_amount']);plt.title('Compliant discount_amount')
plt.figure();plt.hist(trainNonCompliant['clean_up_cost']);plt.title('Compliant clean_up_cost')
plt.figure();plt.hist(trainNonCompliant['judgment_amount']);plt.title('Compliant judgment_amount')

plt.figure(); plt.boxplot([trainCompliant['fine_amount'], trainNonCompliant['fine_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['admin_fee'], trainNonCompliant['admin_fee']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['state_fee'], trainNonCompliant['state_fee']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['late_fee'], trainNonCompliant['late_fee']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['discount_amount'], trainNonCompliant['discount_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['clean_up_cost'], trainNonCompliant['clean_up_cost']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['judgment_amount'], trainNonCompliant['judgment_amount']], whis='range')

train = train.drop(['admin_fee', 'state_fee', 'clean_up_cost'], axis=1)
test = test.drop(['admin_fee', 'state_fee', 'clean_up_cost'], axis=1)

plt.figure(); plt.boxplot([trainCompliant['fine_amount'], trainNonCompliant['fine_amount'], test['fine_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['late_fee'], trainNonCompliant['late_fee'], test['late_fee']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['discount_amount'], trainNonCompliant['discount_amount'], test['discount_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['judgment_amount'], trainNonCompliant['judgment_amount'], test['judgment_amount']], whis='range')

train[pd.isnull(train['hearing_date'])]['compliance'].sum()/len(train[pd.isnull(train['hearing_date'])])
# 0.7312775330396476, 73% of NA hearing dates are compliant! Why and what does not having a hearing date mean?

train['ticket_issued_date'] = pd.to_datetime(train['ticket_issued_date'])
train['hearing_date'] = pd.to_datetime(train['hearing_date'])

train['time_to_hearing'] = train['hearing_date']-train['ticket_issued_date']

trainNullHearingDates = train[pd.isnull(train['hearing_date'])]

# Explore these null dates
plt.figure(); plt.boxplot([trainCompliant['fine_amount'], trainNonCompliant['fine_amount'], test['fine_amount'], trainNullHearingDates['fine_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['late_fee'], trainNonCompliant['late_fee'], test['late_fee'], trainNullHearingDates['late_fee']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['discount_amount'], trainNonCompliant['discount_amount'], test['discount_amount'], trainNullHearingDates['discount_amount']], whis='range')
plt.figure(); plt.boxplot([trainCompliant['judgment_amount'], trainNonCompliant['judgment_amount'], test['judgment_amount'], trainNullHearingDates['judgment_amount']], whis='range')

trainNonNullHearingDates = train[~pd.isnull(train['hearing_date'])]
#trainNonNullHearingDates = trainNonNullHearingDates.reset_index()
trainNonNullHearingDates['time_to_hearing'] = trainNonNullHearingDates['hearing_date']-trainNonNullHearingDates['ticket_issued_date']

trainNonNullHearingDatesCompliant = trainNonNullHearingDates[trainNonNullHearingDates['compliance'] == 1]
trainNonNullHearingDatesNonCompliant = trainNonNullHearingDates[trainNonNullHearingDates['compliance'] == 0]

plt.figure(); plt.boxplot([trainNonNullHearingDatesCompliant['time_to_hearing'], trainNonNullHearingDatesNonCompliant['time_to_hearing']], whis='range')

# Deal with negative time_to_hearing, it's likely a typo
trainNonNullHearingDates[(trainNonNullHearingDates['time_to_hearing']<= pd.Timedelta('0 days 00:00:00')) & (trainNonNullHearingDates['compliance']==1)]

trainPosTimeToHearing = trainNonNullHearingDates[trainNonNullHearingDates['time_to_hearing'] > pd.Timedelta('0 days 00:00:00')]
trainPosTimeToHearing

# Hearing Date information from ticket issued date information


train['violation_street_number'] = pd.Series(train['violation_street_number'], dtype="category")
train['mailing_address_str_number'] = pd.Series(train['mailing_address_str_number'], dtype="category")

train[['judgment_amount', 'compliance']].groupby('compliance').describe()
train[train['judgment_amount'] <= 5000][['judgment_amount', 'compliance']].boxplot(by='compliance')
train[train['judgment_amount'] <= 5000][['judgment_amount', 'compliance']].groupby('compliance').describe()
           
train[train['judgment_amount'] > 5000]


'''
train[['agency_name', 'compliance']].groupby('compliance').describe()
train[['inspector_name', 'compliance']].groupby('compliance').describe()
train[['violator_name', 'compliance']].groupby('compliance').describe()
train[['violation_street_number', 'compliance']].groupby('compliance').describe()
train[['violation_street_name', 'compliance']].groupby('compliance').describe()
train[['mailing_address_str_number', 'compliance']].groupby('compliance').describe()
train[['mailing_address_str_name', 'compliance']].groupby('compliance').describe()
train[['city', 'compliance']].groupby('compliance').describe()
train[['state', 'compliance']].groupby('compliance').describe()
train[['zip_code', 'compliance']].groupby('compliance').describe()
train[['country', 'compliance']].groupby('compliance').describe()
train[['ticket_issued_date', 'compliance']].groupby('compliance').describe()
train[['hearing_date', 'compliance']].groupby('compliance').describe()
train[['violation_code', 'compliance']].groupby('compliance').describe()
train[['violation_description', 'compliance']].groupby('compliance').describe()
train[['disposition', 'compliance']].groupby('compliance').describe()
train[['fine_amount', 'compliance']].groupby('compliance').describe()
train[['late_fee', 'compliance']].groupby('compliance').describe()
train[['discount_amount', 'compliance']].groupby('compliance').describe()
train[['judgment_amount', 'compliance']].groupby('compliance').describe()

test['agency_name'].describe()
test['inspector_name'].describe()
test['violator_name'].describe()
test['violation_street_number'].describe()
test['violation_street_name'].describe()
test['mailing_address_str_number'].describe()
test['mailing_address_str_name'].describe()
test['city'].describe()
test['state'].describe()
test['zip_code'].describe()
test['country'].describe()
test['ticket_issued_date'].describe()
test['hearing_date'].describe()
test['violation_code'].describe()
test['violation_description'].describe()
test['disposition'].describe()
test['fine_amount'].describe()
test['late_fee'].describe()
test['discount_amount'].describe()
test['judgment_amount'].describe()


Training data:
    
[
'agency_name',                  
'inspector_name', 
'violator_name',
'violation_street_number', 
'violation_street_name',       
'mailing_address_str_number', 
'mailing_address_str_name', 
'city',
'state', 
'zip_code', 
'country', 
'ticket_issued_date', 
'hearing_date', 
'violation_code', 
'violation_description', 
'disposition', 
'fine_amount',
'late_fee', 
'discount_amount', 
'judgment_amount', 
'compliance'
]

# Testing Data
[
'agency_name',                  # unique 3
'inspector_name', 
'violator_name',
'violation_street_number', 
'violation_street_name',       
'mailing_address_str_number', 
'mailing_address_str_name', 
'city',
'state', 
'zip_code', 
'country', 
'ticket_issued_date', 
'hearing_date', 
'violation_code', 
'violation_description', 
'disposition', 
'fine_amount',
'late_fee', 
'discount_amount', 
'judgment_amount', 
'compliance'
]


'''












 
# Various models to improve upon
model1 = False
model2 = False
model3 = False

if model1:
    # Build an EXTREMELY simple model to start
    train = train.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 'index', 
                        'violator_name', 'hearing_date'], axis=1)
    test = test.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 
                        'violator_name', 'hearing_date'], axis=1)
    train_corr = train[['fine_amount', 'admin_fee', 'state_fee', 
                       'late_fee', 'discount_amount', 'clean_up_cost',
                       'judgment_amount']]
    
    # Simple is going to use the following columns
    # 'fine_amount', 'judgment_amount', 'agency_name', 
    # 'ticket_issued_date', 'disposition'
    
    trainSimple = train[['fine_amount', 'judgment_amount', 'agency_name', 
                         'ticket_issued_date', 'disposition']]

    testSimple = test[['fine_amount', 'judgment_amount', 'agency_name', 
                       'ticket_issued_date', 'disposition']]
    
    trainSimple['ticket_issued_date'] = pd.to_datetime(trainSimple['ticket_issued_date'])
    #trainSimple['year'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).year
    trainSimple['month'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).month
    trainSimple['weekday'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).weekday    
    trainSimple['hour'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).hour
               
    trainSimple['ticket_issued_deadofnight'] = (trainSimple['hour'] >= 0) & (trainSimple['hour'] < 6)
    trainSimple['ticket_issued_morning'] = (trainSimple['hour'] >= 6) & (trainSimple['hour'] < 12)
    trainSimple['ticket_issued_afternoon'] = (trainSimple['hour'] >= 12) & (trainSimple['hour'] < 18)
    trainSimple['ticket_issued_night'] = trainSimple['hour'] >= 18
    #trainSimple['year'] = trainSimple['year'].astype(str)
    trainSimple['month'] = trainSimple['month'].astype(str)
    trainSimple['weekday'] = trainSimple['weekday'].astype(str)

    trainSimple = trainSimple.drop(['ticket_issued_date', 'hour'], axis=1)
               
    df_dummy = pd.get_dummies(trainSimple[['disposition', 'agency_name', 
                                           #'year', 
                                           'month', 'weekday']])
    
    trainSimpleFull = trainSimple.join(df_dummy)
    trainSimpleFull = trainSimpleFull.drop(['disposition', 'agency_name', 
                                          #'year', 
                                          'month', 'weekday'], axis=1)
    
    
    ## Deal with train/test incompatible agency_name
    
    # Unique Train = ['Buildings, Safety Engineering & Env Department', 
    #                 'Health Department',          # Remove
    #                 'Department of Public Works',
    #                 'Detroit Police Department', 
    #                 'Neighborhood City Halls']    # Remove
    
    ## Deal with train/test incompatible disposition
    
    # Unique Train = ['Responsible by Default', 
    #                 'Responsible by Determination',
    #                 'Responsible by Admission', 
    #                 'Responsible (Fine Waived) by Deter'] # Combine as Fee Waived    
    
    trainSimpleFull = trainSimpleFull.drop(['agency_name_Health Department',
                                            'agency_name_Neighborhood City Halls'], axis=1)       
    trainSimpleFull = trainSimpleFull.rename(columns = {'disposition_Responsible (Fine Waived) by Deter':
                                                        'disposition_Responsible (Fine Waived)'})           
                      
    X_train, X_test, y_train, y_test = train_test_split(trainSimpleFull, train['compliance'], random_state=0)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    param_grid = {"max_depth": [5, 10, 15, 20, None]}
    grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)

    clfOpt = RandomForestClassifier(n_estimators=50, random_state=0, 
                                    max_depth=grid_clf_auc.best_params_['max_depth'])
    clfOpt.fit(X_train, y_train)
    
    #print('Accuracy of RF classifier on training set: {:.2f}'.format(clfOpt.score(X_train, y_train)))
    #print('Accuracy of RF classifier on test set: {:.2f}'.format(clfOpt.score(X_test, y_test)))
    print('Cross-validation (AUC)', cross_val_score(clfOpt, trainSimpleFull, train['compliance'], cv=5, scoring = 'roc_auc'))
    print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

    ####################################
    ### Predicting Actual Test dataset
    ####################################
    
    testSimple['ticket_issued_date'] = pd.to_datetime(testSimple['ticket_issued_date'])
    #testSimple['year'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).year
    testSimple['month'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).month
    testSimple['weekday'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).weekday    
    testSimple['hour'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).hour
               
    testSimple['ticket_issued_deadofnight'] = (testSimple['hour'] >= 0) & (testSimple['hour'] < 6)
    testSimple['ticket_issued_morning'] = (testSimple['hour'] >= 6) & (testSimple['hour'] < 12)
    testSimple['ticket_issued_afternoon'] = (testSimple['hour'] >= 12) & (testSimple['hour'] < 18)
    testSimple['ticket_issued_night'] = testSimple['hour'] >= 18
    #testSimple['year'] = testSimple['year'].astype(str)
    testSimple['month'] = testSimple['month'].astype(str)
    testSimple['weekday'] = testSimple['weekday'].astype(str)

    testSimple = testSimple.drop(['ticket_issued_date', 'hour'], axis=1)
               
    df_dummy = pd.get_dummies(testSimple[['disposition', 'agency_name', 
                                           #'year', 
                                           'month', 'weekday']])
    
    testSimpleFull = testSimple.join(df_dummy)
    testSimpleFull = testSimpleFull.drop(['disposition', 'agency_name', 
                                          #'year', 
                                          'month', 'weekday'], axis=1)
    
    ## Deal with train/test incompatible agency_name
    
    # Unique Test = [ 'Department of Public Works',
    #                 'Buildings, Safety Engineering & Env Department',
    #                 'Detroit Police Department']    
    
    ## Deal with train/test incompatible disposition
    
    # Unique Test = ['Responsible by Default', 
    #                'Responsible by Determination',
    #                'Responsible by Admission', 
    #                'Responsible (Fine Waived) by Deter', # Combine as Fee Waived
    #                'Responsible (Fine Waived) by Admis', # Combine as Fee Waived
    #                'Responsible - Compl/Adj by Default', # Remove
    #                'Responsible - Compl/Adj by Determi', # Remove
    #                'Responsible by Dismissal']            # Remove 

    testSimpleFull['disposition_Responsible (Fine Waived)'] = testSimpleFull['disposition_Responsible (Fine Waived) by Deter'] | testSimpleFull['disposition_Responsible (Fine Waived) by Admis']
                   
    testSimpleFull = testSimpleFull.drop(['disposition_Responsible - Compl/Adj by Default',
                                          'disposition_Responsible - Compl/Adj by Determi',
                                          'disposition_Responsible by Dismissal',
                                          'disposition_Responsible (Fine Waived) by Deter',
                                          'disposition_Responsible (Fine Waived) by Admis'], axis=1)       
     
    
    test['compliance'] = clfOpt.predict_proba(testSimpleFull)[:,0]


elif model2:
    # Using GBM
    train = train.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 'index', 
                        'violator_name', 'hearing_date'], axis=1)
    test = test.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 
                        'violator_name', 'hearing_date'], axis=1)
    train_corr = train[['fine_amount', 'admin_fee', 'state_fee', 
                       'late_fee', 'discount_amount', 'clean_up_cost',
                       'judgment_amount']]
    
    # Simple is going to use the following columns
    # 'fine_amount', 'judgment_amount', 'agency_name', 
    # 'ticket_issued_date', 'disposition'
    
    trainSimple = train[['fine_amount', 'judgment_amount', 'agency_name', 
                         'ticket_issued_date', 'disposition']]

    testSimple = test[['fine_amount', 'judgment_amount', 'agency_name', 
                       'ticket_issued_date', 'disposition']]
    
    trainSimple['ticket_issued_date'] = pd.to_datetime(trainSimple['ticket_issued_date'])
    #trainSimple['year'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).year
    trainSimple['month'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).month
    trainSimple['weekday'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).weekday    
    trainSimple['hour'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).hour
               
    trainSimple['ticket_issued_deadofnight'] = (trainSimple['hour'] >= 0) & (trainSimple['hour'] < 6)
    trainSimple['ticket_issued_morning'] = (trainSimple['hour'] >= 6) & (trainSimple['hour'] < 12)
    trainSimple['ticket_issued_afternoon'] = (trainSimple['hour'] >= 12) & (trainSimple['hour'] < 18)
    trainSimple['ticket_issued_night'] = trainSimple['hour'] >= 18
    #trainSimple['year'] = trainSimple['year'].astype(str)
    trainSimple['month'] = trainSimple['month'].astype(str)
    trainSimple['weekday'] = trainSimple['weekday'].astype(str)

    trainSimple = trainSimple.drop(['ticket_issued_date', 'hour'], axis=1)
               
    df_dummy = pd.get_dummies(trainSimple[['disposition', 'agency_name', 
                                           #'year', 
                                           'month', 'weekday']])
    
    trainSimpleFull = trainSimple.join(df_dummy)
    trainSimpleFull = trainSimpleFull.drop(['disposition', 'agency_name', 
                                          #'year', 
                                          'month', 'weekday'], axis=1)
    
    
    ## Deal with train/test incompatible agency_name
    
    # Unique Train = ['Buildings, Safety Engineering & Env Department', 
    #                 'Health Department',          # Remove
    #                 'Department of Public Works',
    #                 'Detroit Police Department', 
    #                 'Neighborhood City Halls']    # Remove
    
    ## Deal with train/test incompatible disposition
    
    # Unique Train = ['Responsible by Default', 
    #                 'Responsible by Determination',
    #                 'Responsible by Admission', 
    #                 'Responsible (Fine Waived) by Deter'] # Combine as Fee Waived    
    
    trainSimpleFull = trainSimpleFull.drop(['agency_name_Health Department',
                                            'agency_name_Neighborhood City Halls'], axis=1)       
    trainSimpleFull = trainSimpleFull.rename(columns = {'disposition_Responsible (Fine Waived) by Deter':
                                                        'disposition_Responsible (Fine Waived)'})           
                      
    X_train, X_test, y_train, y_test = train_test_split(trainSimpleFull, train['compliance'], random_state=0)
    
    #clf = GradientBoostingClassifier(learning_rate=0.1, 
    #                                 min_samples_split=500,
    #                                 min_samples_leaf=50,
    #                                 max_depth=8,
    #                                 max_features='sqrt',
    #                                 subsample=0.8, 
    #                                 random_state=0)
    #param_grid = {'n_estimators':range(90,191,10)}
    #grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
    #grid_clf_auc.fit(X_train, y_train)
    #print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    # Grid best parameter (max. AUC):  {'n_estimators': 120}
    # Grid best score (AUC):  0.797072211488

    #clf2 = GradientBoostingClassifier(n_estimators=120,
    #                                 learning_rate=0.1, 
    #                                 max_features='sqrt', 
    #                                 subsample=0.8,
    #                                 random_state=0)
    #param_grid2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
    #grid_clf_auc2 = GridSearchCV(clf2, param_grid=param_grid2, scoring = 'roc_auc', cv=5)
    #grid_clf_auc2.fit(X_train, y_train)
    #print('Grid best parameter (max. AUC): ', grid_clf_auc2.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc2.best_score_)
    # Grid best parameter (max. AUC):  {'max_depth': 9, 'min_samples_split': 1000}
    # Grid best score (AUC):  0.798955639234
    
    #clf3 = GradientBoostingClassifier(n_estimators=120,
    #                                  max_depth=9,     
    #                                 learning_rate=0.1, 
    #                                 max_features='sqrt', 
    #                                 subsample=0.8,
    #                                 random_state=0)
    #param_grid3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
    #grid_clf_auc3 = GridSearchCV(clf3, param_grid=param_grid3, scoring = 'roc_auc', cv=5)
    #grid_clf_auc3.fit(X_train, y_train)
    #print('Grid best parameter (max. AUC): ', grid_clf_auc3.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc3.best_score_)
    #Grid best parameter (max. AUC):  {'min_samples_leaf': 30, 'min_samples_split': 1800}
    #Grid best score (AUC):  0.798228401366
 
    #clf4 = GradientBoostingClassifier(n_estimators=120,
    #                                  max_depth=9,
    #                                  min_samples_split = 30,  
    #                                  min_samples_leaf = 1800,    
    #                                 learning_rate=0.1, 
    #                                 subsample=0.8,
    #                                 random_state=0)
    #param_grid4 = {'max_features':range(7,20,2)}
    #grid_clf_auc4 = GridSearchCV(clf4, param_grid=param_grid4, scoring = 'roc_auc', cv=5)
    #grid_clf_auc4.fit(X_train, y_train)
    #print('Grid best parameter (max. AUC): ', grid_clf_auc4.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc4.best_score_) 
    #Grid best parameter (max. AUC):  {'max_features': 19}
    #Grid best score (AUC):  0.792773797124
    
    clf5 = GradientBoostingClassifier(n_estimators=120,
                                      max_depth=9,
                                      min_samples_split = 30,  
                                      min_samples_leaf = 1800,  
                                      max_features = 19,
                                     learning_rate=0.1, 
                                     random_state=0)
    param_grid5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    grid_clf_auc5 = GridSearchCV(clf5, param_grid=param_grid5, scoring = 'roc_auc', cv=5)
    grid_clf_auc5.fit(X_train, y_train)
    print('Grid best parameter (max. AUC): ', grid_clf_auc5.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc5.best_score_)   
    
    ## Indepdent increase and decrease of n_estimators and learning_rate
    
    #clf6 = GradientBoostingClassifier(n_estimators=200,
    #                                  max_depth=####,
    #                                  min_samples_split = ####,  
    #                                  min_samples_leaf = ####,  
    #                                  max_features = ####,
    #                                  subsample = ####,
    #                                 learning_rate=0.05, 
    #                                 random_state=0)
    #clf6.fit(X_train, y_train)

    #print('Cross-validation (AUC)', cross_val_score(clfOpt, trainSimpleFull, train['compliance'], cv=5, scoring = 'roc_auc'))
    #print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

    
    
    '''
    clfOpt = RandomForestClassifier(n_estimators=50, random_state=0, 
                                    max_depth=grid_clf_auc.best_params_['max_depth'])
    clfOpt.fit(X_train, y_train)
    
    #print('Accuracy of RF classifier on training set: {:.2f}'.format(clfOpt.score(X_train, y_train)))
    #print('Accuracy of RF classifier on test set: {:.2f}'.format(clfOpt.score(X_test, y_test)))
    print('Cross-validation (AUC)', cross_val_score(clfOpt, trainSimpleFull, train['compliance'], cv=5, scoring = 'roc_auc'))
    print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

    ####################################
    ### Predicting Actual Test dataset
    ####################################
    
    testSimple['ticket_issued_date'] = pd.to_datetime(testSimple['ticket_issued_date'])
    #testSimple['year'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).year
    testSimple['month'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).month
    testSimple['weekday'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).weekday    
    testSimple['hour'] = pd.DatetimeIndex(testSimple['ticket_issued_date']).hour
               
    testSimple['ticket_issued_deadofnight'] = (testSimple['hour'] >= 0) & (testSimple['hour'] < 6)
    testSimple['ticket_issued_morning'] = (testSimple['hour'] >= 6) & (testSimple['hour'] < 12)
    testSimple['ticket_issued_afternoon'] = (testSimple['hour'] >= 12) & (testSimple['hour'] < 18)
    testSimple['ticket_issued_night'] = testSimple['hour'] >= 18
    #testSimple['year'] = testSimple['year'].astype(str)
    testSimple['month'] = testSimple['month'].astype(str)
    testSimple['weekday'] = testSimple['weekday'].astype(str)

    testSimple = testSimple.drop(['ticket_issued_date', 'hour'], axis=1)
               
    df_dummy = pd.get_dummies(testSimple[['disposition', 'agency_name', 
                                           #'year', 
                                           'month', 'weekday']])
    
    testSimpleFull = testSimple.join(df_dummy)
    testSimpleFull = testSimpleFull.drop(['disposition', 'agency_name', 
                                          #'year', 
                                          'month', 'weekday'], axis=1)
    
    ## Deal with train/test incompatible agency_name
    
    # Unique Test = [ 'Department of Public Works',
    #                 'Buildings, Safety Engineering & Env Department',
    #                 'Detroit Police Department']    
    
    ## Deal with train/test incompatible disposition
    
    # Unique Test = ['Responsible by Default', 
    #                'Responsible by Determination',
    #                'Responsible by Admission', 
    #                'Responsible (Fine Waived) by Deter', # Combine as Fee Waived
    #                'Responsible (Fine Waived) by Admis', # Combine as Fee Waived
    #                'Responsible - Compl/Adj by Default', # Remove
    #                'Responsible - Compl/Adj by Determi', # Remove
    #                'Responsible by Dismissal']            # Remove 

    testSimpleFull['disposition_Responsible (Fine Waived)'] = testSimpleFull['disposition_Responsible (Fine Waived) by Deter'] | testSimpleFull['disposition_Responsible (Fine Waived) by Admis']
                   
    testSimpleFull = testSimpleFull.drop(['disposition_Responsible - Compl/Adj by Default',
                                          'disposition_Responsible - Compl/Adj by Determi',
                                          'disposition_Responsible by Dismissal',
                                          'disposition_Responsible (Fine Waived) by Deter',
                                          'disposition_Responsible (Fine Waived) by Admis'], axis=1)       
     
    
    test['compliance'] = clfOpt.predict_proba(testSimpleFull)[:,0]
    '''

    
elif model3:
    train = train.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 'index', 
                        'violator_name', 'hearing_date'], axis=1)
    test = test.drop(['mailing_address_str_number', 'mailing_address_str_name', 
                        'city','state', 'zip_code', 'country', 
                        'violator_name', 'hearing_date'], axis=1)
    train_corr = train[['fine_amount', 'admin_fee', 'state_fee', 
                       'late_fee', 'discount_amount', 'clean_up_cost',
                       'judgment_amount']]
    
    # Simple is going to use the following columns
    # 'fine_amount', 'judgment_amount', 'agency_name', 'inspector_name', 
    # 'ticket_issued_date', 'violation_code', 'violation_description',
    # 'disposition'
    
    trainSimple = train[['fine_amount', 'judgment_amount', 'agency_name', 
                         'inspector_name', 'ticket_issued_date', 'disposition',
                         'violation_code', 'violation_description',]]

    testSimple = test[['fine_amount', 'judgment_amount', 'agency_name', 
                         'inspector_name', 'ticket_issued_date', 'disposition',
                         'violation_code', 'violation_description',]]
    
    trainSimple['ticket_issued_date'] = pd.to_datetime(trainSimple['ticket_issued_date'])
    trainSimple['year'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).year
    trainSimple['month'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).month
    trainSimple['weekday'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).weekday    
    trainSimple['hour'] = pd.DatetimeIndex(trainSimple['ticket_issued_date']).hour
               
    trainSimple['ticket_issued_deadofnight'] = (trainSimple['hour'] >= 0) & (trainSimple['hour'] < 6)
    trainSimple['ticket_issued_morning'] = (trainSimple['hour'] >= 6) & (trainSimple['hour'] < 12)
    trainSimple['ticket_issued_afternoon'] = (trainSimple['hour'] >= 12) & (trainSimple['hour'] < 18)
    trainSimple['ticket_issued_night'] = trainSimple['hour'] >= 18
    trainSimple = trainSimple.drop(['ticket_issued_date', 'hour'], axis=1)
               
    df_dummy = pd.get_dummies(trainSimple[['inspector_name', 'disposition', 
                                           'violation_code', 'violation_description',
                                           'year', 'month', 'weekday']])
    
    trainSimpleFull = trainSimple.join(df_dummy)
    trainSimpleFull = trainSimpleFull.drop(['inspector_name', 'disposition', 
                                           'violation_code', 'violation_description',
                                           'year', 'month', 'weekday'], axis=1)
        

###################################
###### End of If Statement ########
###################################

# pd.DataFrame(test[['ticket_id','compliance']]).to_csv('test_file1.csv', sep=',')

###################################


'''
['agency_name', 'inspector_name', 'violator_name',
       'violation_street_number', 'violation_street_name',
       'mailing_address_str_number', 'mailing_address_str_name', 'city',
       'state', 'zip_code', 'country', 'ticket_issued_date', 'hearing_date',
       'violation_code', 'violation_description', 'disposition', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'compliance']
'''


'''country_ind = train.columns.get_loc('country')

ind = train[train['city'].astype(str).str.contains('CANA')].index            
for index in ind :
    train.iloc[index,country_ind] = 'Cana'
    
ind = train[train['city'].astype(str).str.contains('AUSTR')].index            
for index in ind :
    train.iloc[index,country_ind] = 'Aust'
      
ind = train[train['city'].astype(str).str.contains('ENGLAND')].index            
for index in ind :
    train.iloc[index,country_ind] = 'England'   

ind = train[train['city'].astype(str).str.contains('TAIWAN')].index            
for index in ind :
    train.iloc[index,country_ind] = 'Taiwan'  
              
ind = train[train['city'].astype(str).str.contains('SINGAPORE')].index            
for index in ind :
    train.iloc[index,country_ind] = 'Singapore'  '''              

#train[train['city'].astype(str).str.contains('AUSTR')].replace('USA', 'Aust', inplace=True)
#train[train['city'].astype(str).str.contains('TAIWAN')][['state', 'city', 'country']]
#train[pd.isnull(train['state'])][['state', 'city', 'country']]
#train[(train['state']) == 'MT'][['state', 'city', 'country']]























