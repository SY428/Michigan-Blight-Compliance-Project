#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:56:55 2017

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

def diff_seconds(date2, date1):
   res =  (date2.dt.date - date1.dt.date).dt.total_seconds()
   res += (date2.dt.hour - date1.dt.hour) * 3600
   res += (date2.dt.minute - date1.dt.minute) * 60
   res += date2.dt.second - date1.dt.second
   return res


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

# Separate the disposition with fees waived from fees not waived
trainFeeWaived = train3[train3['disposition'].astype(str).str.contains('Fine Waived')]
trainFeeNotWaived = train3[~train3['disposition'].astype(str).str.contains('Fine Waived')]

# Before transforming the data,make sure it's compatible with those of testing data

######################################################
### Deal with testing data
######################################################
# Basic feature transformations
test['ticket_issued_date'] = pd.to_datetime(test['ticket_issued_date'])
test['hearing_date'] = pd.to_datetime(test['hearing_date'])
test['time_to_hearing'] = test['hearing_date']-test['ticket_issued_date']

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
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('[0123456789]', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('[-_]', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(' ', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('.', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('`', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(';', 'L'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace(',', ''))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SUOTH', 'SOUTH'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOUTFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOTUHFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOUTJFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SIOUTHFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOOTHFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOUTHFILELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SUTHFIELD', 'SOUTHFIELD'))
test1['city'] = test1['city'].astype(str).map(lambda x: str(x).replace('SOOUTHFIELD', 'SOUTHFIELD'))

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
test1.iloc[test1[test1['city']=='DETR'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROTIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DETROIY'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='FETROIT'].index, test1.columns == 'city'] = 'DETROIT'
test1.iloc[test1[test1['city']=='DERROIT'].index, test1.columns == 'city'] = 'DETROIT'

test1.iloc[test1[test1['city']=='LASANGELAS'].index, test1.columns == 'city'] = 'LOSANGELES'
test1.iloc[test1[test1['city']=='LOSANGLES'].index, test1.columns == 'city'] = 'LOSANGELES'
test1.iloc[test1[test1['city']=='LASVAGAS'].index, test1.columns == 'city'] = 'LASVEGAS'

test1.iloc[test1[test1['city']=='SANDIEEGO'].index, test1.columns == 'city'] = 'SANDIEGO'
          
test1.iloc[test1[test1['city']=='BERLKEY'].index, test1.columns == 'city'] = 'BERKLEY'

test1.iloc[test1[test1['city']=='SINGAPROE'].index, test1.columns == 'city'] = 'SINGAPORE'

test1.iloc[test1[test1['city']=='SANFRANICISCO'].index, test1.columns == 'city'] = 'SANFRANCISCO'

test1.iloc[test1[test1['city']=='STERINGHEIGHTS'].index, test1.columns == 'city'] = 'STERLINGHEIGHTS'

test1.iloc[test1[test1['city']=='CLEVLAND'].index, test1.columns == 'city'] = 'CLEVELAND'

test1.iloc[test1[test1['city']=='BLOOOMFIELDHILLS'].index, test1.columns == 'city'] = 'BLOOMFIELDHILLS'
test1.iloc[test1[test1['city']=='BLOOMFIELDSHILLS'].index, test1.columns == 'city'] = 'BLOOMFIELDHILLS'

test1.iloc[test1[test1['city']=='COLWNBIA'].index, test1.columns == 'city'] = 'COLUMBIA'

test1.iloc[test1[test1['city']=='NEYYORK'].index, test1.columns == 'city'] = 'NEWYORK'
test1.iloc[test1[test1['city']=='NEWYOUR'].index, test1.columns == 'city'] = 'NEWYORK'
          

# Separate the disposition with fees waived from fees not waived
testFeeWaived = test1[test1['disposition'].astype(str).str.contains('Fine Waived') | test1['disposition'].astype(str).str.contains('Dismissal')]
testFeeNotWaived = test1[~test1['disposition'].astype(str).str.contains('Fine Waived') & ~test1['disposition'].astype(str).str.contains('Dismissal')]

testFeeNotWaived = testFeeNotWaived.reset_index()
# Check categorical data and lable encode transform

######################################################
### Deal with rest of training data
######################################################
#################################################################
### Make non intercepting items the same category for ease of 
### modeling: Training
###
### Variables: 
###    city, violation_code, violation_street_name, 
###    violation_description, zip_code, violation_street_number
#################################################################
trainFeeNotWaived1 = trainFeeNotWaived.copy(deep=True)
trainFeeNotWaived1 = trainFeeNotWaived1.drop('index', axis=1)
trainFeeNotWaived1 = trainFeeNotWaived1.reset_index()

# elements in training data not in testing data
trainlist = [x for x in trainFeeNotWaived['city'].unique() if x not in testFeeNotWaived['city'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['city'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'city'] = 'NaN'

trainlist = [x for x in trainFeeNotWaived['violation_code'].unique() if x not in testFeeNotWaived['violation_code'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['violation_code'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'violation_code'] = 'NaN'

trainlist = [x for x in trainFeeNotWaived['violation_street_name'].unique() if x not in testFeeNotWaived['violation_street_name'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['violation_street_name'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'violation_street_name'] = 'NaN'

trainlist = [x for x in trainFeeNotWaived['violation_description'].unique() if x not in testFeeNotWaived['violation_description'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['violation_description'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'violation_description'] = 'NaN'

trainlist = [x for x in trainFeeNotWaived['zip_code'].unique() if x not in testFeeNotWaived['zip_code'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['zip_code'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'zip_code'] = 'NaN'

trainlist = [x for x in trainFeeNotWaived['violation_street_number'].unique() if x not in testFeeNotWaived['violation_street_number'].unique()]
trainFeeNotWaived1.iloc[trainFeeNotWaived1[trainFeeNotWaived1['violation_street_number'].isin(trainlist)].index,trainFeeNotWaived1.columns == 'violation_street_number'] = 'NaN'


#################################################################
### Make non intercepting items the same category for ease of 
### modeling: Testing
###
### Variables: 
###    city, violation_code, violation_street_name, 
###    violation_description, zip_code, violation_street_number
#################################################################
testFeeNotWaived1 = testFeeNotWaived.copy(deep=True)
#testFeeNotWaived1 = testFeeNotWaived1.drop('index', axis=1)
testFeeNotWaived1 = testFeeNotWaived1.reset_index()

# elements in testing data not in training data
testlist = [x for x in testFeeNotWaived['city'].unique() if x not in trainFeeNotWaived['city'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['city'].isin(testlist)].index,testFeeNotWaived1.columns == 'city'] = 'NaN'

testlist = [x for x in testFeeNotWaived['violation_code'].unique() if x not in trainFeeNotWaived['violation_code'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['violation_code'].isin(testlist)].index,testFeeNotWaived1.columns == 'violation_code'] = 'NaN'

testlist = [x for x in testFeeNotWaived['violation_street_name'].unique() if x not in trainFeeNotWaived['violation_street_name'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['violation_street_name'].isin(testlist)].index,testFeeNotWaived1.columns == 'violation_street_name'] = 'NaN'

testlist = [x for x in testFeeNotWaived['violation_description'].unique() if x not in trainFeeNotWaived['violation_description'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['violation_description'].isin(testlist)].index,testFeeNotWaived1.columns == 'violation_description'] = 'NaN'

testlist = [x for x in testFeeNotWaived['zip_code'].unique() if x not in trainFeeNotWaived['zip_code'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['zip_code'].isin(testlist)].index,testFeeNotWaived1.columns == 'zip_code'] = 'NaN'

testlist = [x for x in testFeeNotWaived['violation_street_number'].unique() if x not in trainFeeNotWaived['violation_street_number'].unique()]
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['violation_street_number'].isin(testlist)].index,testFeeNotWaived1.columns == 'violation_street_number'] = 'NaN'

[x for x in testFeeNotWaived1['disposition'].unique() if x not in trainFeeNotWaived1['disposition'].unique()]
testFeeNotWaived1[['disposition', 'state_fee']].groupby('disposition').agg(['mean', 'count']).sort_values(by=[('state_fee', 'count')], ascending=False)
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['disposition'].astype(str).str.contains('Default')].index, testFeeNotWaived1.columns == 'disposition'] = 'Responsible by Default'
testFeeNotWaived1.iloc[testFeeNotWaived1[testFeeNotWaived1['disposition'].astype(str).str.contains('Determi')].index, testFeeNotWaived1.columns == 'disposition'] = 'Responsible by Determination'

###############################################################################
### le transform
###############################################################################

#le1 = preprocessing.LabelEncoder()
#le2 = preprocessing.LabelEncoder()
le3 = preprocessing.LabelEncoder()
#le4 = preprocessing.LabelEncoder()
le5 = preprocessing.LabelEncoder()
le6 = preprocessing.LabelEncoder()
#le7 = preprocessing.LabelEncoder()
#le8 = preprocessing.LabelEncoder()
#le9 = preprocessing.LabelEncoder()
le10 = preprocessing.LabelEncoder()
le11 = preprocessing.LabelEncoder()
le12 = preprocessing.LabelEncoder()
#le13 = preprocessing.LabelEncoder()
le14 = preprocessing.LabelEncoder()

#trainFeeNotWaived['agency_name'] = le1.fit_transform(trainFeeNotWaived['agency_name'])
#trainFeeNotWaived['country'] = le2.fit_transform(trainFeeNotWaived['country'])
trainFeeNotWaived1['disposition'] = le3.fit_transform(trainFeeNotWaived1['disposition'])
#trainFeeNotWaived['inspector_name'] = le4.fit_transform(trainFeeNotWaived['inspector_name'])
trainFeeNotWaived1['city'] = le5.fit_transform(trainFeeNotWaived1['city'])
trainFeeNotWaived1['violation_code'] = le6.fit_transform(trainFeeNotWaived1['violation_code'])
#trainFeeNotWaived['violator_name'] = le7.fit_transform(trainFeeNotWaived['violator_name'])
#trainFeeNotWaived['mailing_address_str_name'] = le8.fit_transform(trainFeeNotWaived['mailing_address_str_name'])
#trainFeeNotWaived['state'] = le9.fit_transform(trainFeeNotWaived['state'])
trainFeeNotWaived1['violation_street_name'] = le10.fit_transform(trainFeeNotWaived1['violation_street_name'])
trainFeeNotWaived1['violation_description'] = le11.fit_transform(trainFeeNotWaived1['violation_description'])
trainFeeNotWaived1['zip_code'] = le12.fit_transform(trainFeeNotWaived1['zip_code'])
#trainFeeNotWaived['mailing_address_str_number'] = le13.fit_transform(trainFeeNotWaived['mailing_address_str_number'])
trainFeeNotWaived1['violation_street_number'] = le14.fit_transform(trainFeeNotWaived1['violation_street_number'])

# trainFeeNotWaived1['time_to_hearing'] = trainFeeNotWaived1['time_to_hearing']/ np.timedelta64(1, 's')
trainFeeNotWaived1['hearing_date'] = pd.to_datetime(trainFeeNotWaived1['hearing_date'].fillna('1900-01-01 00:00:00'))

trainFeeNotWaived1['time_to_hearing'] = diff_seconds(trainFeeNotWaived1['hearing_date'], trainFeeNotWaived1['ticket_issued_date'])

trainFeeNotWaived1 = trainFeeNotWaived1.drop('index', axis=1)

#trainHasHearingDate = trainFeeNotWaived1[~pd.isnull(trainFeeNotWaived1['hearing_date'])]
trainHasHearingDate = trainFeeNotWaived1[trainFeeNotWaived1['hearing_date'] != pd.to_datetime('1900-01-01 00:00:00')]
#trainDNHHearingDate = trainFeeNotWaived1[pd.isnull(trainFeeNotWaived1['hearing_date'])]
trainDNHHearingDate = trainFeeNotWaived1[trainFeeNotWaived1['hearing_date'] == pd.to_datetime('1900-01-01 00:00:00')]


# Remove the features deemed too low on the importance scale

''' Remove from trainHasHearingDate
admin_fee, state_fee, clean_up_cost, country, address_BOX, address_MILE, agency_name, state, fine_amount, people, ticket_issued_date, hearing_date
# Remove due to high number of difference between training and testing data
inspector_name, violator_name, mailing_address_str_name, mailing_address_str_number
'''
trainHasHearingDate = trainHasHearingDate.drop(['admin_fee', 'state_fee', 'clean_up_cost', 
                                                'country', 'address_BOX', 'address_MILE', 
                                                'agency_name', 'state', 'fine_amount', 'people',
                                                'ticket_issued_date', 'hearing_date',
                                                'inspector_name', 'violator_name', 'mailing_address_str_name', 'mailing_address_str_number'], axis=1)

''' Remove from trainDNHHearingDate
address_MILE, country, clean_up_cost, admin_fee, state_fee, address_BOX, state, agency_name, people, ticket_issued_date, hearing_date, time_to_hearing
# Remove due to high number of difference between training and testing data
inspector_name, violator_name, mailing_address_str_name, mailing_address_str_number
'''

trainDNHHearingDate = trainDNHHearingDate.drop(['address_MILE', 'country', 'clean_up_cost', 
                                                'admin_fee', 'state_fee', 'address_BOX', 
                                                'state', 'agency_name', 'people',
                                                'ticket_issued_date', 'hearing_date', 'time_to_hearing',
                                                'inspector_name', 'violator_name', 'mailing_address_str_name', 'mailing_address_str_number'], axis=1)
    
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

######################################################
### GradientBoostingClassifier
######################################################

clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.9, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

GBCFull_clfOpt = clfOpt

######################################################
### LogisticRegression
######################################################

clfOpt = LogisticRegression(penalty='l1',C=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

LogRegFull_clfOpt = clfOpt

### On null hearing date dataset

clfOpt = LogisticRegression(penalty='l1',C=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train_sp, y_train_sp, cv=5, scoring = 'roc_auc'))
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test_sp, y_test_sp, cv=5, scoring = 'roc_auc'))

LogRegNull_clfOpt = clfOpt

######################################################
### xgboost: Extreme Gradient Boosting
######################################################

##################################################################################################################################################################

######################################################
### Deal with testing data part 2
######################################################

testFeeNotWaived1['disposition'] = le3.transform(testFeeNotWaived1['disposition'])
testFeeNotWaived1['city'] = le5.transform(testFeeNotWaived1['city'])
testFeeNotWaived1['violation_code'] = le6.transform(testFeeNotWaived1['violation_code'])
testFeeNotWaived1['violation_street_name'] = le10.transform(testFeeNotWaived1['violation_street_name'])
testFeeNotWaived1['violation_description'] = le11.transform(testFeeNotWaived1['violation_description'])
testFeeNotWaived1['zip_code'] = le12.transform(testFeeNotWaived1['zip_code'])
testFeeNotWaived1['violation_street_number'] = le14.transform(testFeeNotWaived1['violation_street_number'])

#testFeeNotWaived1['time_to_hearing'] = testFeeNotWaived1['time_to_hearing']/ np.timedelta64(1, 's')
testFeeNotWaived1['hearing_date'] = pd.to_datetime(testFeeNotWaived1['hearing_date'].fillna('1900-01-01 00:00:00'))

testFeeNotWaived1['time_to_hearing'] = diff_seconds(testFeeNotWaived1['hearing_date'], testFeeNotWaived1['ticket_issued_date'])


testFeeNotWaived1 = testFeeNotWaived1.drop('index', axis=1)

#testHasHearingDate = testFeeNotWaived1[~pd.isnull(testFeeNotWaived1['hearing_date'])]
testHasHearingDate = testFeeNotWaived1[testFeeNotWaived1['hearing_date'] != pd.to_datetime('1900-01-01 00:00:00')]
#testDNHHearingDate = testFeeNotWaived1[pd.isnull(testFeeNotWaived1['hearing_date'])]
testDNHHearingDate = testFeeNotWaived1[testFeeNotWaived1['hearing_date'] == pd.to_datetime('1900-01-01 00:00:00')]

# Remove the features deemed too low on the importance scale

''' Remove from testHasHearingDate
admin_fee, state_fee, clean_up_cost, country, address_BOX, address_MILE, agency_name, state, fine_amount, people, ticket_issued_date, hearing_date
# Remove due to high number of difference between training and testing data
inspector_name, violator_name, mailing_address_str_name, mailing_address_str_number
'''
testHasHearingDate = testHasHearingDate.drop(['admin_fee', 'state_fee', 'clean_up_cost', 
                                                'country', 'level_0',
                                                'agency_name', 'state', 'fine_amount', 
                                                'ticket_issued_date', 'hearing_date',
                                                'inspector_name', 'violator_name', 'mailing_address_str_name', 'mailing_address_str_number'], axis=1)

''' Remove from testDNHHearingDate
address_MILE, country, clean_up_cost, admin_fee, state_fee, address_BOX, state, agency_name, people, ticket_issued_date, hearing_date, time_to_hearing
# Remove due to high number of difference between training and testing data
inspector_name, violator_name, mailing_address_str_name, mailing_address_str_number
'''

testDNHHearingDate = testDNHHearingDate.drop(['country', 'clean_up_cost', 
                                                'admin_fee', 'state_fee', 
                                                'state', 'agency_name', 'level_0',
                                                'ticket_issued_date', 'hearing_date', 'time_to_hearing',
                                                'inspector_name', 'violator_name', 'mailing_address_str_name', 'mailing_address_str_number'], axis=1)

######################################################
### Predict Testing data with hearing dates
######################################################

### Logarithmic Regression
LogRegFull_clfOpt.fit(trainHasHearingDate.iloc[:,trainHasHearingDate.columns != 'compliance'], trainHasHearingDate['compliance'])
LogRegFull_PredProb = LogRegFull_clfOpt.predict_proba(testHasHearingDate.iloc[:,testHasHearingDate.columns != 'ticket_id'])[:,1]

### Gradient Boosting Classifier
GBCFull_clfOpt.fit(trainHasHearingDate.iloc[:,trainHasHearingDate.columns != 'compliance'], trainHasHearingDate['compliance'])
GBCFull_PredProb = GBCFull_clfOpt.predict_proba(testHasHearingDate.iloc[:,testHasHearingDate.columns != 'ticket_id'])[:,1]

######################################################
### Predict Testing data with NO hearing dates
######################################################

### Logarithmic Regression
LogRegNull_clfOpt.fit(trainDNHHearingDate.iloc[:,trainDNHHearingDate.columns != 'compliance'], trainDNHHearingDate['compliance'])
LogRegNull_PredProb = LogRegNull_clfOpt.predict_proba(testDNHHearingDate.iloc[:,testDNHHearingDate.columns != 'ticket_id'])[:,1]

######################################################
### Predict Testing data with fines waived
######################################################

### Combine test data

testP1 = testFeeWaived.copy(deep=True)
testP1['compliance'] = 1
testP1 = testP1[['ticket_id','compliance']]     
      
testP2 = testHasHearingDate.copy(deep=True)
testP2['compliance'] = np.mean(np.array([LogRegFull_PredProb, GBCFull_PredProb]), axis=0)
testP2 = testP2[['ticket_id','compliance']]

testP3 = testDNHHearingDate.copy(deep=True)
testP3['compliance'] = LogRegNull_PredProb
testP3 = testP3[['ticket_id','compliance']]     

testFinal = testP1.copy(deep=True)
testFinal = testFinal.append(testP2)
testFinal = testFinal.append(testP3)

     
      