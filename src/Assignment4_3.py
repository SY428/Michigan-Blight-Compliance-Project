#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:53:22 2017

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
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics

train = pd.read_csv('downloads/train.csv', encoding = 'ISO-8859-1')
test = pd.read_csv('downloads/test.csv')
addresses = pd.read_csv('downloads/addresses.csv')
latlons = pd.read_csv('downloads/latlons.csv')
    
# Clean up training data
train = train.drop(['payment_amount', 'payment_date', 'payment_status', 
                    'balance_due', 'collection_status', 'compliance_detail'], axis=1)
trainOrig = train.copy(deep=True)
train = train[~np.isnan(train['compliance'])]
train = train.reset_index()

# Remove columns with too many NaNs
train = train.drop(['violation_zip_code', 'non_us_str_code', 'grafitti_status'], axis=1)
test = test.drop(['violation_zip_code', 'non_us_str_code', 'grafitti_status'], axis=1)

train = train.drop('ticket_id', axis=1)

# Basic feature transformations
train['ticket_issued_date'] = pd.to_datetime(train['ticket_issued_date'])
train['hearing_date'] = pd.to_datetime(train['hearing_date'])
train['time_to_hearing'] = train['hearing_date']-train['ticket_issued_date']

# Get the number of NaN per column
pd.isnull(train.iloc[:,:]).sum()
pd.isnull(test.iloc[:,:]).sum()

# Label Encode categorical variables
# agency_name, country, disposition, inspector_name, city, violation_code, violation_description, violation_street_name
# NaNs: violator_name, mailing_address_str_number, mailing_address_str_name, state, zip_code
train.iloc[train[pd.isnull(train['violator_name'])]['violator_name'].index,train.columns == 'violator_name'] = 'NaN'
train.iloc[train[pd.isnull(train['mailing_address_str_number'])]['mailing_address_str_number'].index,train.columns == 'mailing_address_str_number'] = 'NaN'
train.iloc[train[pd.isnull(train['mailing_address_str_name'])]['mailing_address_str_name'].index,train.columns == 'mailing_address_str_name'] = 'NaN'
train.iloc[train[pd.isnull(train['state'])]['state'].index,train.columns == 'state'] = 'NaN'
train.iloc[train[pd.isnull(train['zip_code'])]['zip_code'].index,train.columns == 'zip_code'] = 'NaN'

train['violation_street_number'] = train['violation_street_number'].astype(str)
train['mailing_address_str_number'] = train['mailing_address_str_number'].astype(str)
train['zip_code'] = train['zip_code'].astype(str)

train1 = train.copy(deep=True)

le = preprocessing.LabelEncoder()

train1['agency_name'] = le.fit_transform(train1['agency_name'])
train1['country'] = le.fit_transform(train1['country'])
train1['disposition'] = le.fit_transform(train1['disposition'])
train1['inspector_name'] = le.fit_transform(train1['inspector_name'])
train1['city'] = le.fit_transform(train1['city'])
train1['violation_code'] = le.fit_transform(train1['violation_code'])
train1['violator_name'] = le.fit_transform(train1['violator_name'])
train1['mailing_address_str_name'] = le.fit_transform(train1['mailing_address_str_name'])
train1['state'] = le.fit_transform(train1['state'])
train1['violation_street_name'] = le.fit_transform(train1['violation_street_name'])
train1['violation_description'] = le.fit_transform(train1['violation_description'])
train1['zip_code'] = le.fit_transform(train1['zip_code'])
train1['mailing_address_str_number'] = le.fit_transform(train1['mailing_address_str_number'])
train1['violation_street_number'] = le.fit_transform(train1['violation_street_number'])
train1['time_to_hearing'] = train1['time_to_hearing']/ np.timedelta64(1, 's')

train1 = train1.drop('index', axis=1)

trainHasHearingDate = train1[~pd.isnull(train1['hearing_date'])]
trainDNHHearingDate = train1[pd.isnull(train1['hearing_date'])]

# Fit a few simple models to grab feature importance
RFC = RandomForestClassifier(n_estimators=100, random_state=0).fit(trainHasHearingDate.iloc[:,(trainHasHearingDate.columns != 'compliance') & (trainHasHearingDate.columns != 'ticket_issued_date') & (trainHasHearingDate.columns != 'hearing_date')], trainHasHearingDate['compliance'])
RFCFeatImp = pd.DataFrame(trainHasHearingDate.iloc[:,(trainHasHearingDate.columns != 'compliance') & (trainHasHearingDate.columns != 'ticket_issued_date') & (trainHasHearingDate.columns != 'hearing_date')].columns, RFC.feature_importances_)
RFCFeatImp = RFCFeatImp.sort_index(ascending=False)

RFC1 = RandomForestClassifier(n_estimators=100, random_state=0).fit(trainDNHHearingDate.iloc[:,(trainDNHHearingDate.columns != 'compliance') & (trainDNHHearingDate.columns != 'ticket_issued_date') & (trainDNHHearingDate.columns != 'hearing_date') & (trainDNHHearingDate.columns != 'time_to_hearing')], trainDNHHearingDate['compliance'])
RFCFeatImp1 = pd.DataFrame(trainDNHHearingDate.iloc[:,(trainDNHHearingDate.columns != 'compliance') & (trainDNHHearingDate.columns != 'ticket_issued_date') & (trainDNHHearingDate.columns != 'hearing_date') & (trainDNHHearingDate.columns != 'time_to_hearing')].columns, RFC1.feature_importances_)
RFCFeatImp1 = RFCFeatImp1.sort_index(ascending=False)

###################################################
### Feature Extraction
###################################################

train2 = train.copy(deep=True)

# From violator_name find if violator is a company or individual
violator_name_index = np.array([])

temp_index = train2[train2['violator_name'].astype(str).str.contains('INC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Ii]nc.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('GROUP')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('LLC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('MANAGEMENT')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('RESTAURANT')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('LIMITED')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('CORP')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('ASSOC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('INDUSTRIAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('CO.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('COMMERCIAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('HOSPITAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('CLEANER')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Cc][Hh][Uu][Rr][Cc][Hh]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('INTERNATIONAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Ss]ervices')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Cc]ommunity')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Cc]enter')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('CENTER')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('ISLAND')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Pp]artnership')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('L.L.C.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[\!\@\#\$\%\^\&\*\(\)\'\"\/\?\<\>\~\+\=]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[0123456789]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train2[train2['violator_name'].astype(str).str.contains('[Bb][Aa][Nn][Kk]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)

violator_name_index = np.unique(violator_name_index)
violator_name_index = violator_name_index.astype(int)

train2['people'] = 1
for index in violator_name_index:
    train2.iloc[index,train2.columns == 'people']  = 0

# From mailing_address_str_name find if address is a PO box
mailing_address_str_name_index = np.array([])

temp_index = train2[train2['mailing_address_str_name'].astype(str).str.contains('[Bb][Oo][Xx]')]['mailing_address_str_name'].index
mailing_address_str_name_index = np.append(mailing_address_str_name_index, temp_index)
mailing_address_str_name_index = mailing_address_str_name_index.astype(int)

train2['address_BOX'] = 0
for index in mailing_address_str_name_index:
    train2.iloc[index,train2.columns == 'address_BOX']  = 1
               
# From mailing_address_str_name find if address is a Mile
mailing_address_str_name_index = np.array([])
temp_index = train2[train2['mailing_address_str_name'].astype(str).str.contains('[Mm][Ii][Ll][Ee]')]['mailing_address_str_name'].index
mailing_address_str_name_index = np.append(mailing_address_str_name_index, temp_index)
mailing_address_str_name_index = mailing_address_str_name_index.astype(int)

train2['address_MILE'] = 0
for index in mailing_address_str_name_index:
    train2.iloc[index,train2.columns == 'address_MILE']  = 1
               
# Deal with too many cities
train2['city'] = train2['city'].str.upper()
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace(' ', ''))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('.', ''))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('`', ''))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace(';', 'L'))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace(',', ''))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('SUOTH', 'SOUTH'))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('SOUTFIELD', 'SOUTHFIELD'))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('SOTUHFIELD', 'SOUTHFIELD'))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('SOUTJFIELD', 'SOUTHFIELD'))
train2['city'] = train2['city'].astype(str).map(lambda x: str(x).replace('BUFFLAO', 'BUFFALO'))
### Note WOW people really like to spell Detroit incorrectly!!!
train2.iloc[train2[train2['city']=='DEROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DERTOIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETRIUT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROITDETROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DTEROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROIOT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DTROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='ETROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETRIOT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROTI'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETEROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROI'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='CETROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROITF'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROOIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETORIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETRORIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROIT1'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETROITQ'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETRROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETEOIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DERTROIT'].index, train2.columns == 'city'] = 'DETROIT'
train2.iloc[train2[train2['city']=='DETRTOIT'].index, train2.columns == 'city'] = 'DETROIT'

train2.iloc[train2[train2['city']=='LOSANGLES'].index, train2.columns == 'city'] = 'LOSANGELES'
train2.iloc[train2[train2['city']=='LASVAGAS'].index, train2.columns == 'city'] = 'LASVEGAS'

# Check for every violation code what is the percentage of compliance
train2[['violation_code', 'compliance']].groupby('violation_code').mean().sort_values(by='compliance', ascending=False)
train2[['violation_description', 'compliance']].groupby('violation_description').mean().sort_values(by='compliance', ascending=False)

# Check disposition
train2[['disposition', 'compliance']].groupby('disposition').mean().sort_values(by='compliance', ascending=False)
# Fine waived had a 100% chance of compliance!

# Separate the disposition with fees waived from fees not waived
trainFeeWaived = train2[train2['disposition'].astype(str).str.contains('Fine Waived')]
trainFeeNotWaived = train2[~train2['disposition'].astype(str).str.contains('Fine Waived')]

# le transform for feature importance check
trainFeeNotWaived['agency_name'] = le.fit_transform(trainFeeNotWaived['agency_name'])
trainFeeNotWaived['country'] = le.fit_transform(trainFeeNotWaived['country'])
trainFeeNotWaived['disposition'] = le.fit_transform(trainFeeNotWaived['disposition'])
trainFeeNotWaived['inspector_name'] = le.fit_transform(trainFeeNotWaived['inspector_name'])
trainFeeNotWaived['city'] = le.fit_transform(trainFeeNotWaived['city'])
trainFeeNotWaived['violation_code'] = le.fit_transform(trainFeeNotWaived['violation_code'])
trainFeeNotWaived['violator_name'] = le.fit_transform(trainFeeNotWaived['violator_name'])
trainFeeNotWaived['mailing_address_str_name'] = le.fit_transform(trainFeeNotWaived['mailing_address_str_name'])
trainFeeNotWaived['state'] = le.fit_transform(trainFeeNotWaived['state'])
trainFeeNotWaived['violation_street_name'] = le.fit_transform(trainFeeNotWaived['violation_street_name'])
trainFeeNotWaived['violation_description'] = le.fit_transform(trainFeeNotWaived['violation_description'])
trainFeeNotWaived['zip_code'] = le.fit_transform(trainFeeNotWaived['zip_code'])
trainFeeNotWaived['mailing_address_str_number'] = le.fit_transform(trainFeeNotWaived['mailing_address_str_number'])
trainFeeNotWaived['violation_street_number'] = le.fit_transform(trainFeeNotWaived['violation_street_number'])
trainFeeNotWaived['time_to_hearing'] = trainFeeNotWaived['time_to_hearing']/ np.timedelta64(1, 's')

trainFeeNotWaived = trainFeeNotWaived.drop('index', axis=1)

trainHasHearingDate = trainFeeNotWaived[~pd.isnull(trainFeeNotWaived['hearing_date'])]
trainDNHHearingDate = trainFeeNotWaived[pd.isnull(trainFeeNotWaived['hearing_date'])]

# Fit a few simple models to grab feature importance
RFC = RandomForestClassifier(n_estimators=100, random_state=0).fit(trainHasHearingDate.iloc[:,(trainHasHearingDate.columns != 'compliance') & (trainHasHearingDate.columns != 'ticket_issued_date') & (trainHasHearingDate.columns != 'hearing_date')], trainHasHearingDate['compliance'])
RFCFeatImp = pd.DataFrame(trainHasHearingDate.iloc[:,(trainHasHearingDate.columns != 'compliance') & (trainHasHearingDate.columns != 'ticket_issued_date') & (trainHasHearingDate.columns != 'hearing_date')].columns, RFC.feature_importances_)
RFCFeatImp = RFCFeatImp.sort_index(ascending=False)

RFC1 = RandomForestClassifier(n_estimators=100, random_state=0).fit(trainDNHHearingDate.iloc[:,(trainDNHHearingDate.columns != 'compliance') & (trainDNHHearingDate.columns != 'ticket_issued_date') & (trainDNHHearingDate.columns != 'hearing_date') & (trainDNHHearingDate.columns != 'time_to_hearing')], trainDNHHearingDate['compliance'])
RFCFeatImp1 = pd.DataFrame(trainDNHHearingDate.iloc[:,(trainDNHHearingDate.columns != 'compliance') & (trainDNHHearingDate.columns != 'ticket_issued_date') & (trainDNHHearingDate.columns != 'hearing_date') & (trainDNHHearingDate.columns != 'time_to_hearing')].columns, RFC1.feature_importances_)
RFCFeatImp1 = RFCFeatImp1.sort_index(ascending=False)

###################################################
### Final check on categorical data
###################################################

train3 = train.copy(deep=True)

# From violator_name find if violator is a company or individual
violator_name_index = np.array([])

temp_index = train3[train3['violator_name'].astype(str).str.contains('INC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Ii]nc.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('GROUP')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('LLC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('MANAGEMENT')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('RESTAURANT')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('LIMITED')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('CORP')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('ASSOC')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('INDUSTRIAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('CO.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('COMMERCIAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('HOSPITAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('CLEANER')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Cc][Hh][Uu][Rr][Cc][Hh]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('INTERNATIONAL')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Ss]ervices')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Cc]ommunity')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Cc]enter')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('CENTER')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('ISLAND')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Pp]artnership')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('L.L.C.')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[\!\@\#\$\%\^\&\*\(\)\'\"\/\?\<\>\~\+\=]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[0123456789]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)
temp_index = train3[train3['violator_name'].astype(str).str.contains('[Bb][Aa][Nn][Kk]')]['violator_name'].index
violator_name_index = np.append(violator_name_index, temp_index)

violator_name_index = np.unique(violator_name_index)
violator_name_index = violator_name_index.astype(int)

train3['people'] = 1
for index in violator_name_index:
    train3.iloc[index,train3.columns == 'people']  = 0

# From mailing_address_str_name find if address is a PO box
mailing_address_str_name_index = np.array([])

temp_index = train3[train3['mailing_address_str_name'].astype(str).str.contains('[Bb][Oo][Xx]')]['mailing_address_str_name'].index
mailing_address_str_name_index = np.append(mailing_address_str_name_index, temp_index)
mailing_address_str_name_index = mailing_address_str_name_index.astype(int)

train3['address_BOX'] = 0
for index in mailing_address_str_name_index:
    train3.iloc[index,train3.columns == 'address_BOX']  = 1
               
# From mailing_address_str_name find if address is a Mile
mailing_address_str_name_index = np.array([])
temp_index = train3[train3['mailing_address_str_name'].astype(str).str.contains('[Mm][Ii][Ll][Ee]')]['mailing_address_str_name'].index
mailing_address_str_name_index = np.append(mailing_address_str_name_index, temp_index)
mailing_address_str_name_index = mailing_address_str_name_index.astype(int)

train3['address_MILE'] = 0
for index in mailing_address_str_name_index:
    train3.iloc[index,train3.columns == 'address_MILE']  = 1
               
# Deal with too many cities
train3['city'] = train3['city'].str.upper()
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace(' ', ''))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('.', ''))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('`', ''))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace(';', 'L'))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace(',', ''))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('SUOTH', 'SOUTH'))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('SOUTFIELD', 'SOUTHFIELD'))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('SOTUHFIELD', 'SOUTHFIELD'))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('SOUTJFIELD', 'SOUTHFIELD'))
train3['city'] = train3['city'].astype(str).map(lambda x: str(x).replace('BUFFLAO', 'BUFFALO'))
### Note WOW people really like to spell Detroit incorrectly!!!
train3.iloc[train3[train3['city']=='DEROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DERTOIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETRIUT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROITDETROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DTEROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROIOT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DTROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='ETROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETRIOT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROTI'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETEROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROI'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='CETROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROITF'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROOIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETORIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETRORIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROIT1'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETROITQ'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETRROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETEOIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DERTROIT'].index, train3.columns == 'city'] = 'DETROIT'
train3.iloc[train3[train3['city']=='DETRTOIT'].index, train3.columns == 'city'] = 'DETROIT'

train3.iloc[train3[train3['city']=='LOSANGLES'].index, train3.columns == 'city'] = 'LOSANGELES'
train3.iloc[train3[train3['city']=='LASVAGAS'].index, train3.columns == 'city'] = 'LASVEGAS'

'''
['index', 'agency_name', 'inspector_name', 'violator_name',
       'violation_street_number', 'violation_street_name',
       'mailing_address_str_number', 'mailing_address_str_name', 'city',
       'state', 'zip_code', 'country', 'ticket_issued_date', 'hearing_date',
       'violation_code', 'violation_description', 'disposition', 'fine_amount',
       'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
       'clean_up_cost', 'judgment_amount', 'compliance', 'time_to_hearing',
       'people', 'address_BOX', 'address_MILE']
'''

train3[['agency_name', 'compliance']].groupby('agency_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['inspector_name', 'compliance']].groupby('inspector_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_street_number', 'compliance']].groupby('violation_street_number').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_street_name', 'compliance']].groupby('violation_street_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['mailing_address_str_number', 'compliance']].groupby('mailing_address_str_number').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['city', 'compliance']].groupby('city').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['state', 'compliance']].groupby('state').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['zip_code', 'compliance']].groupby('zip_code').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['country', 'compliance']].groupby('country').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_code', 'compliance']].groupby('violation_code').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_description', 'compliance']].groupby('violation_description').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['disposition', 'compliance']].groupby('disposition').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['people', 'compliance']].groupby('people').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['address_BOX', 'compliance']].groupby('address_BOX').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['address_MILE', 'compliance']].groupby('address_MILE').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)

# Separate the disposition with fees waived from fees not waived
trainFeeWaived = train3[train3['disposition'].astype(str).str.contains('Fine Waived')]
trainFeeNotWaived = train3[~train3['disposition'].astype(str).str.contains('Fine Waived')]

le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le3 = preprocessing.LabelEncoder()
le4 = preprocessing.LabelEncoder()
le5 = preprocessing.LabelEncoder()
le6 = preprocessing.LabelEncoder()
le7 = preprocessing.LabelEncoder()
le8 = preprocessing.LabelEncoder()
le9 = preprocessing.LabelEncoder()
le10 = preprocessing.LabelEncoder()
le11 = preprocessing.LabelEncoder()
le12 = preprocessing.LabelEncoder()
le13 = preprocessing.LabelEncoder()
le14 = preprocessing.LabelEncoder()

# le transform for feature importance check
trainFeeNotWaived['agency_name'] = le1.fit_transform(trainFeeNotWaived['agency_name'])
trainFeeNotWaived['country'] = le2.fit_transform(trainFeeNotWaived['country'])
trainFeeNotWaived['disposition'] = le3.fit_transform(trainFeeNotWaived['disposition'])
trainFeeNotWaived['inspector_name'] = le4.fit_transform(trainFeeNotWaived['inspector_name'])
trainFeeNotWaived['city'] = le5.fit_transform(trainFeeNotWaived['city'])
trainFeeNotWaived['violation_code'] = le6.fit_transform(trainFeeNotWaived['violation_code'])
trainFeeNotWaived['violator_name'] = le7.fit_transform(trainFeeNotWaived['violator_name'])
trainFeeNotWaived['mailing_address_str_name'] = le8.fit_transform(trainFeeNotWaived['mailing_address_str_name'])
trainFeeNotWaived['state'] = le9.fit_transform(trainFeeNotWaived['state'])
trainFeeNotWaived['violation_street_name'] = le10.fit_transform(trainFeeNotWaived['violation_street_name'])
trainFeeNotWaived['violation_description'] = le11.fit_transform(trainFeeNotWaived['violation_description'])
trainFeeNotWaived['zip_code'] = le12.fit_transform(trainFeeNotWaived['zip_code'])
trainFeeNotWaived['mailing_address_str_number'] = le13.fit_transform(trainFeeNotWaived['mailing_address_str_number'])
trainFeeNotWaived['violation_street_number'] = le14.fit_transform(trainFeeNotWaived['violation_street_number'])
trainFeeNotWaived['time_to_hearing'] = trainFeeNotWaived['time_to_hearing']/ np.timedelta64(1, 's')

trainFeeNotWaived = trainFeeNotWaived.drop('index', axis=1)

trainHasHearingDate = trainFeeNotWaived[~pd.isnull(trainFeeNotWaived['hearing_date'])]
trainDNHHearingDate = trainFeeNotWaived[pd.isnull(trainFeeNotWaived['hearing_date'])]

# Remove the features deemed too low on the importance scale

''' Remove from trainHasHearingDate
admin_fee, state_fee, clean_up_cost, country, address_BOX, address_MILE, agency_name, state, fine_amount, people, ticket_issued_date, hearing_date
'''
trainHasHearingDate = trainHasHearingDate.drop(['admin_fee', 'state_fee', 'clean_up_cost', 
                                                'country', 'address_BOX', 'address_MILE', 
                                                'agency_name', 'state', 'fine_amount', 'people',
                                                'ticket_issued_date', 'hearing_date'], axis=1)

''' Remove from trainDNHHearingDate
address_MILE, country, clean_up_cost, admin_fee, state_fee, address_BOX, state, agency_name, people, ticket_issued_date, hearing_date, time_to_hearing
'''

trainDNHHearingDate = trainDNHHearingDate.drop(['address_MILE', 'country', 'clean_up_cost', 
                                                'admin_fee', 'state_fee', 'address_BOX', 
                                                'state', 'agency_name', 'people',
                                                'ticket_issued_date', 'hearing_date', 'time_to_hearing'], axis=1)

############################################################################################################
### Training data: split into 3 categories
### 1. Fine waived          <-- no need to predict (special treatment during test stage )
### 2. No Hearing Dates     <-- need to predict
### 3. The rest             <-- need to predict
###
### Models: RandomForest Classifier, SVC, Logistic Regression, GradientBoosting, Xgboost
############################################################################################################

# Train trainHasHearingDate first
X_train, X_test, y_train, y_test = train_test_split(trainHasHearingDate.iloc[:,trainHasHearingDate.columns != 'compliance'], trainHasHearingDate['compliance'], random_state=0)

X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(trainDNHHearingDate.iloc[:,trainDNHHearingDate.columns != 'compliance'], trainDNHHearingDate['compliance'], random_state=0)
X_train_sp1, X_test_sp1, y_train_sp1, y_test_sp1 = train_test_split(trainDNHHearingDate.iloc[:,trainDNHHearingDate.columns != 'compliance'], trainDNHHearingDate['compliance'], random_state=1)

######################################################
### GradientBoostingClassifier
######################################################

clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0, verbose=1)
param_grid = {'n_estimators':range(90,191,10)}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

### n_estimators
clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.82953866  0.8268176   0.81567553  0.8269606   0.81970842]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80676038  0.81069061  0.81472087  0.80135898  0.79493629]

# min_samples_split
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=250, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=750, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [0.82879649  0.82548146  0.81579744  0.82426755  0.81662662]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.81170064  0.8078384   0.8140242   0.80220234  0.79861999]

# min_samples_leaf
clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=75, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

# max_depth
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=15, max_features='sqrt', subsample=0.8, random_state=0)
clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=20, max_features='sqrt', subsample=0.8, random_state=0)
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=18, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.8394054   0.83606562  0.82226907  0.83333175  0.82849829]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80636486  0.81720089  0.8200595   0.79809802  0.79412746]

# max_features
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=20, max_features='auto', subsample=0.8, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

# subsample
#clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=850, min_samples_leaf=100, max_depth=20, max_features='sqrt', subsample=0.7, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

# learning_rate
clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.05, min_samples_split=850, min_samples_leaf=100, max_depth=20, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.83678924  0.82973772  0.81931808  0.83197378  0.82508957]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.81159336  0.8188392   0.81453644  0.79922481  0.80193127]

clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.001, min_samples_split=850, min_samples_leaf=100, max_depth=20, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.8211559   0.8184931   0.81064868  0.82019283  0.80842908]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80461428  0.80860842  0.81670645  0.7963745   0.79779787]

### On null hearing date dataset

### n_estimators
# clfOpt = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
#clfOpt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, min_samples_split=200, min_samples_leaf=10, max_depth=4, max_features='sqrt', subsample=0.8, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train_sp1, y_train_sp1, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test_sp1, y_test_sp1, cv=5, scoring = 'roc_auc'))
# This is horrible!

######################################################
### SVC: Support Vector Classifier
######################################################

clf = SVC(kernel='rbf')
param_grid = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
clfOpt = SVC(kernel='rbf', gamma=grid_clf_auc.best_params_['gamma'], random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

clf = SVC(kernel='rbf', gamma=0.1)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

clfOpt = SVC(kernel='rbf', gamma=grid_clf_auc.best_params_['C'], random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

clfOpt = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

clfOpt = SVC(kernel='linear', C=1, gamma=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))



######################################################
### LogisticRegression
######################################################

clf = LogisticRegression(penalty='l2', random_state=0)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
clfOpt = LogisticRegression(penalty='l2', C=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.52479264  0.52962085  0.556731    0.55405839  0.5392378 ]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.51463778  0.55377348  0.57606039  0.57906547  0.5288347 ]

clf = LogisticRegression(penalty='l1', random_state=0)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
#clfOpt = LogisticRegression(penalty='l1',C=10)
clfOpt = LogisticRegression(penalty='l1',C=50, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.77433842  0.77664385  0.76250951  0.78298697  0.75505223]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.7646598   0.77614817  0.79055496  0.75449432  0.76784363]

### On null hearing date dataset

clf = LogisticRegression(penalty='l1', random_state=0)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100, 500, 1000]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train_sp, y_train_sp)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
clfOpt = LogisticRegression(penalty='l1',C=10, random_state=0)
#clfOpt = LogisticRegression(penalty='l1',C=5, random_state=0)
#clfOpt = LogisticRegression(penalty='l1',C=7, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train_sp, y_train_sp, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test_sp, y_test_sp, cv=5, scoring = 'roc_auc'))

######################################################
### xgboost: Extreme Gradient Boosting
######################################################

xgbTrain = xgb.DMatrix(data=X_train, label=y_train)

xgb_param = {'base_score': y_train.mean(),
 'colsample_bylevel': 1,
 'colsample_bytree': 0.8,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 5,
 'min_child_weight': 1,
 'missing': None,
 'n_estimators': 1000,
 'nthread': 4,
 'objective': 'binary:logistic',
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': 27,
 'silent': 1,
 'subsample': 0.8}

cvresult = xgb.cv(params=xgb_param, dtrain=xgbTrain, num_boost_round=1000, nfold=5, metrics='auc', verbose_eval=True)

#xgb = XGBClassifier()
#xgb.set_params(n_estimators=cvresult.shape[0])

########### Using GridSearch instead

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, silent=1), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=0)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# {'max_depth': 9, 'min_child_weight': 1} : 0.83703714799806117
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch1.predict_proba(X_train)[:,1]))
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch1.predict_proba(X_test)[:,1]))


# Go a little closer to the data
param_test2 = {
 'max_depth':[8,9,10],
 'min_child_weight':[1,2,3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5,
                                                  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
'''
{'max_depth': 9, 'min_child_weight': 1}, : 0.83703714799806117
'''

##################################################################################################################################################################

##################################################################################################################################################################

######################################################
### Deal with testing data
######################################################

test.iloc[test[pd.isnull(test['violator_name'])]['violator_name'].index,test.columns == 'violator_name'] = 'NaN'
test.iloc[test[pd.isnull(test['mailing_address_str_number'])]['mailing_address_str_number'].index,test.columns == 'mailing_address_str_number'] = 'NaN'
test.iloc[test[pd.isnull(test['mailing_address_str_name'])]['mailing_address_str_name'].index,test.columns == 'mailing_address_str_name'] = 'NaN'
test.iloc[test[pd.isnull(test['state'])]['state'].index,test.columns == 'state'] = 'NaN'
test.iloc[test[pd.isnull(test['zip_code'])]['zip_code'].index,test.columns == 'zip_code'] = 'NaN'
               
test['violation_street_number'] = test['violation_street_number'].astype(str)
test['mailing_address_str_number'] = test['mailing_address_str_number'].astype(str)
test['zip_code'] = test['zip_code'].astype(str)

test1 = test.copy(deep=True)

pd.isnull(test1.iloc[:,:]).sum()
# Deal with the special NA in city
test1[['city', 'state_fee']].groupby('city').agg(['mean', 'count']).sort_values(by=[('state_fee', 'count')], ascending=False)
test1.iloc[test1[pd.isnull(test1['city'])]['city'].index,test1.columns == 'city'] = 'DETROIT'

# Deal with too many cities
test1['city'] = test1['city'].str.upper()
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(' ', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('.', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('`', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(';', 'L'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(',', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SUOTH', 'SOUTH'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOUTFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOTUHFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOUTJFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('BUFFLAO', 'BUFFALO'))
### Note WOW people really like to spell Detroit incorrectly!!!
test1.iloc[test1[test1['city']=='DEROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DERTOIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETRIUT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROITDETROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DTEROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROIOT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DTROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='ETROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETRIOT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROTI'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETEROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROI'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='CETROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROITF'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROOIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETORIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETRORIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROIT1'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROITQ'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETRROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETEOIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DERTROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETRTOIT'].index, test1.columns == 'city'] = 'DETROIT'

test1.iloc[test1[test1['city']=='LOSANGLES'].index, test1.columns == 'city'] = 'LOSANGELES'
test1.iloc[test1[test1['city']=='LASVAGAS'].index, test1.columns == 'city'] = 'LASVEGAS'

# Separate the disposition with fees waived from fees not waived
testFeeWaived = test1[test1['disposition'].astype(str).str.contains('Fine Waived') or test1['disposition'].astype(str).str.contains('Dismissal')]
testFeeNotWaived = test1[~test1['disposition'].astype(str).str.contains('Fine Waived') & ~test1['disposition'].astype(str).str.contains('Dismissal')]

testFeeNotWaived = testFeeNotWaived.reset_index()
# Check categorical data and lable encode transform

'''
train3[['agency_name', 'compliance']].groupby('agency_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['inspector_name', 'compliance']].groupby('inspector_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'count')], ascending=False)
train3[['violation_street_number', 'compliance']].groupby('violation_street_number').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_street_name', 'compliance']].groupby('violation_street_name').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['mailing_address_str_number', 'compliance']].groupby('mailing_address_str_number').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['city', 'compliance']].groupby('city').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['state', 'compliance']].groupby('state').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['zip_code', 'compliance']].groupby('zip_code').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['country', 'compliance']].groupby('country').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_code', 'compliance']].groupby('violation_code').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['violation_description', 'compliance']].groupby('violation_description').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['disposition', 'compliance']].groupby('disposition').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['people', 'compliance']].groupby('people').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['address_BOX', 'compliance']].groupby('address_BOX').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
train3[['address_MILE', 'compliance']].groupby('address_MILE').agg(['mean', 'count']).sort_values(by=[('compliance', 'mean')], ascending=False)
'''

#testFeeNotWaived['agency_name'] = le1.transform(testFeeNotWaived['agency_name'])

#testFeeNotWaived['country'] = le2.transform(testFeeNotWaived['country'])

[x for x in testFeeNotWaived['disposition'].unique() if x not in train3['disposition'].unique()]
testFeeNotWaived[['disposition', 'state_fee']].groupby('disposition').agg(['mean', 'count']).sort_values(by=[('state_fee', 'count')], ascending=False)
testFeeNotWaived.iloc[testFeeNotWaived[testFeeNotWaived['disposition'].astype(str).str.contains('Default')].index, testFeeNotWaived.columns == 'disposition'] = 'Responsible by Default'
testFeeNotWaived.iloc[testFeeNotWaived[testFeeNotWaived['disposition'].astype(str).str.contains('Determi')].index, testFeeNotWaived.columns == 'disposition'] = 'Responsible by Determination'
testFeeNotWaived['disposition'] = le3.transform(testFeeNotWaived['disposition'])

[x for x in testFeeNotWaived['inspector_name'].unique() if x not in train3['inspector_name'].unique()]
testFeeNotWaived['inspector_name'] = le4.transform(testFeeNotWaived['inspector_name'])

[x for x in testFeeNotWaived['city'].unique() if x not in train3['city'].unique()]
testFeeNotWaived['city'] = le5.transform(testFeeNotWaived['city'])

[x for x in testFeeNotWaived['violation_code'].unique() if x not in train3['violation_code'].unique()]
testFeeNotWaived['violation_code'] = le6.transform(testFeeNotWaived['violation_code'])

[x for x in testFeeNotWaived['violator_name'].unique() if x not in train3['violator_name'].unique()]
testFeeNotWaived['violator_name'] = le7.transform(testFeeNotWaived['violator_name'])

[x for x in testFeeNotWaived['mailing_address_str_name'].unique() if x not in train3['mailing_address_str_name'].unique()]
testFeeNotWaived['mailing_address_str_name'] = le8.transform(testFeeNotWaived['mailing_address_str_name'])

#testFeeNotWaived['state'] = le9.transform(testFeeNotWaived['state'])

[x for x in testFeeNotWaived['violation_street_name'].unique() if x not in train3['violation_street_name'].unique()]
testFeeNotWaived['violation_street_name'] = le10.transform(testFeeNotWaived['violation_street_name'])

[x for x in testFeeNotWaived['violation_description'].unique() if x not in train3['violation_description'].unique()]
testFeeNotWaived['violation_description'] = le11.transform(testFeeNotWaived['violation_description'])

[x for x in testFeeNotWaived['zip_code'].unique() if x not in train3['zip_code'].unique()]
testFeeNotWaived['zip_code'] = le12.transform(testFeeNotWaived['zip_code'])

[x for x in testFeeNotWaived['mailing_address_str_number'].unique() if x not in train3['mailing_address_str_number'].unique()]
testFeeNotWaived['mailing_address_str_number'] = le13.transform(testFeeNotWaived['mailing_address_str_number'])

[x for x in testFeeNotWaived['violation_street_number'].unique() if x not in train3['violation_street_number'].unique()]
testFeeNotWaived['violation_street_number'] = le14.transform(testFeeNotWaived['violation_street_number'])

testFeeNotWaived['time_to_hearing'] = testFeeNotWaived['time_to_hearing']/ np.timedelta64(1, 's')

testFeeNotWaived = testFeeNotWaived.drop('index', axis=1)

testHasHearingDate = testFeeNotWaived[~pd.isnull(testFeeNotWaived['hearing_date'])]
testDNHHearingDate = testFeeNotWaived[pd.isnull(testFeeNotWaived['hearing_date'])]

# Remove the features deemed too low on the importance scale

''' Remove from testHasHearingDate
admin_fee, state_fee, clean_up_cost, country, address_BOX, address_MILE, agency_name, state, fine_amount, people, ticket_issued_date, hearing_date
'''
testHasHearingDate = testHasHearingDate.drop(['admin_fee', 'state_fee', 'clean_up_cost', 
                                                'country', 'address_BOX', 'address_MILE', 
                                                'agency_name', 'state', 'fine_amount', 'people',
                                                'ticket_issued_date', 'hearing_date'], axis=1)

''' Remove from testDNHHearingDate
address_MILE, country, clean_up_cost, admin_fee, state_fee, address_BOX, state, agency_name, people, ticket_issued_date, hearing_date, time_to_hearing
'''

testDNHHearingDate = testDNHHearingDate.drop(['address_MILE', 'country', 'clean_up_cost', 
                                                'admin_fee', 'state_fee', 'address_BOX', 
                                                'state', 'agency_name', 'people',
                                                'ticket_issued_date', 'hearing_date', 'time_to_hearing'], axis=1)









