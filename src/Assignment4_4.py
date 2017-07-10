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

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['inspector_name'].unique() if x not in trainFeeNotWaived['inspector_name'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['inspector_name'].unique() if x not in testFeeNotWaived['inspector_name'].unique()]
len([x for x in trainFeeNotWaived['inspector_name'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.6669129849390989
len([x for x in testFeeNotWaived['inspector_name'] if x in testlist])/testFeeNotWaived.shape[0] # 0.2918728619349696

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['city'].unique() if x not in trainFeeNotWaived['city'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['city'].unique() if x not in testFeeNotWaived['city'].unique()]
len([x for x in trainFeeNotWaived['city'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.02739142687165357
len([x for x in testFeeNotWaived['city'] if x in testlist])/testFeeNotWaived.shape[0] # 0.04528546281842638

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['violation_code'].unique() if x not in trainFeeNotWaived['violation_code'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['violation_code'].unique() if x not in testFeeNotWaived['violation_code'].unique()]
len([x for x in trainFeeNotWaived['violation_code'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.00272411309766102
len([x for x in testFeeNotWaived['violation_code'] if x in testlist])/testFeeNotWaived.shape[0] # 0.0031884154239596135

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['violator_name'].unique() if x not in trainFeeNotWaived['violator_name'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['violator_name'].unique() if x not in testFeeNotWaived['violator_name'].unique()]
len([x for x in trainFeeNotWaived['violator_name'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.8972727557378589
len([x for x in testFeeNotWaived['violator_name'] if x in testlist])/testFeeNotWaived.shape[0] # 0.8818127470191637

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['mailing_address_str_name'].unique() if x not in trainFeeNotWaived['mailing_address_str_name'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['mailing_address_str_name'].unique() if x not in testFeeNotWaived['mailing_address_str_name'].unique()]
len([x for x in trainFeeNotWaived['mailing_address_str_name'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.2691736856937095
len([x for x in testFeeNotWaived['mailing_address_str_name'] if x in testlist])/testFeeNotWaived.shape[0] # 0.33063203693247867
 
# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['violation_street_name'].unique() if x not in trainFeeNotWaived['violation_street_name'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['violation_street_name'].unique() if x not in testFeeNotWaived['violation_street_name'].unique()]
len([x for x in trainFeeNotWaived['violation_street_name'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.013094529855653318
len([x for x in testFeeNotWaived['violation_street_name'] if x in testlist])/testFeeNotWaived.shape[0] # 0.0028230761566309078

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['violation_description'].unique() if x not in trainFeeNotWaived['violation_description'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['violation_description'].unique() if x not in testFeeNotWaived['violation_description'].unique()]
len([x for x in trainFeeNotWaived['violation_description'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.0032063124275918215
len([x for x in testFeeNotWaived['violation_description'] if x in testlist])/testFeeNotWaived.shape[0] # 0.0031385964329602445
   
# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['zip_code'].unique() if x not in trainFeeNotWaived['zip_code'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['zip_code'].unique() if x not in testFeeNotWaived['zip_code'].unique()]
len([x for x in trainFeeNotWaived['zip_code'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.036841281272505244
len([x for x in testFeeNotWaived['zip_code'] if x in testlist])/testFeeNotWaived.shape[0] # 0.06403400976452224
   
# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['mailing_address_str_number'].unique() if x not in trainFeeNotWaived['mailing_address_str_number'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['mailing_address_str_number'].unique() if x not in testFeeNotWaived['mailing_address_str_number'].unique()]
len([x for x in trainFeeNotWaived['mailing_address_str_number'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.9839997495068415
len([x for x in testFeeNotWaived['mailing_address_str_number'] if x in testlist])/testFeeNotWaived.shape[0] # 0..9831777873725465

# elements in training data not in testing data
testlist = [x for x in testFeeNotWaived['violation_street_number'].unique() if x not in trainFeeNotWaived['violation_street_number'].unique()]
# elements in testing data not in training data
trainlist = [x for x in trainFeeNotWaived['violation_street_number'].unique() if x not in testFeeNotWaived['violation_street_number'].unique()]
len([x for x in trainFeeNotWaived['violation_street_number'] if x in trainlist])/trainFeeNotWaived.shape[0] # 0.15611359864733695
len([x for x in testFeeNotWaived['violation_street_number'] if x in testlist])/testFeeNotWaived.shape[0] # 0.03053904148261317

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

trainFeeNotWaived1['time_to_hearing'] = trainFeeNotWaived1['time_to_hearing']/ np.timedelta64(1, 's')

trainFeeNotWaived1 = trainFeeNotWaived1.drop('index', axis=1)

trainHasHearingDate = trainFeeNotWaived1[~pd.isnull(trainFeeNotWaived1['hearing_date'])]
trainDNHHearingDate = trainFeeNotWaived1[pd.isnull(trainFeeNotWaived1['hearing_date'])]


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

clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0, verbose=1)
param_grid = {'n_estimators':range(90,191,10)}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

### n_estimators
clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.8275293   0.81989909  0.81327206  0.82109434  0.81456662]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80280887  0.8100045   0.80857364  0.79232002  0.78766489]

# min_samples_split
#clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=600, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

# min_samples_leaf
#clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=100, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=85, max_depth=8, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.82869589  0.8188273   0.81103995  0.82367149  0.81432422]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80494405  0.80833343  0.8086953   0.79394741  0.78700192]

# max_depth
clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.8, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.82075647  0.81493619  0.80876203  0.81809275  0.80843657]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80203693  0.80821083  0.81115893  0.79440684  0.789631  ]

# max_features
#clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.8, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))

# subsample
clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.9, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.82146074  0.81703606  0.80691402  0.81665675  0.8079239 ]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.80402296  0.80324818  0.81745439  0.79507681  0.79453074]

# learning_rate
clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.9, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.81517433  0.81168295  0.80383715  0.81402449  0.80255611]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.79972303  0.80410389  0.81536105  0.79821231  0.79710464]

GBCFull_clfOpt = clfOpt

### On null hearing date dataset

### n_estimators
#clfOpt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, min_samples_split=500, min_samples_leaf=85, max_depth=5, max_features='sqrt', subsample=0.9, random_state=0)
#print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train_sp, y_train_sp, cv=5, scoring = 'roc_auc'))
#print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test_sp, y_test_sp, cv=5, scoring = 'roc_auc'))
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
clfOpt = LogisticRegression(penalty='l2', C=1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.53197575  0.5191677   0.52845331  0.53153939  0.52124607]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.5099084   0.528176    0.51361208  0.53812218  0.52717621]

clf = LogisticRegression(penalty='l1', random_state=0)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
#clfOpt = LogisticRegression(penalty='l1',C=100, random_state=0)
clfOpt = LogisticRegression(penalty='l1',C=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train, y_train, cv=5, scoring = 'roc_auc'))
# [ 0.77454034  0.77491841  0.76072318  0.78189722  0.76141367]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test, y_test, cv=5, scoring = 'roc_auc'))
# [ 0.76051545  0.77152306  0.78840393  0.75512239  0.76423825]

LogRegFull_clfOpt = clfOpt

### On null hearing date dataset

clf = LogisticRegression(penalty='l1', random_state=0)
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 10, 100, 500, 1000]}
grid_clf_auc = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
grid_clf_auc.fit(X_train_sp, y_train_sp)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
#clfOpt = LogisticRegression(penalty='l1',C=10, random_state=0)
#clfOpt = LogisticRegression(penalty='l1',C=1, random_state=0)
clfOpt = LogisticRegression(penalty='l1',C=0.1, random_state=0)
print('Cross-validation (AUC) on train data', cross_val_score(clfOpt, X_train_sp, y_train_sp, cv=5, scoring = 'roc_auc'))
# [ 0.89328063  0.96086957  0.85217391  0.86956522  0.96086957]
# [ 0.8972332   0.95217391  0.81304348  0.86956522  0.95217391]
# [ 0.87747036  0.94782609  0.82608696  0.86521739  0.94347826]
print('Cross-validation (AUC) on test data', cross_val_score(clfOpt, X_test_sp, y_test_sp, cv=5, scoring = 'roc_auc'))
# [ 0.95        0.88888889  0.88888889  0.94444444  1.        ]
# [ 0.95        0.88888889  1.          1.          1.        ]
# [ 1.          0.88888889  1.          1.          1.        ]

LogRegNull_clfOpt = clfOpt

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

param_test1 = {'max_depth':range(3,15,1)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=5, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, silent=1), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=0)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# {'max_depth': 10}, 0.82944354184504587
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch1.predict_proba(X_train)[:,1])) # 0.95042982815
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch1.predict_proba(X_test)[:,1])) # 0.8297293955


param_test2 = {'min_child_weight':range(1,10,1)}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# {'min_child_weight': 1}, 0.82944354184504587
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch2.predict_proba(X_train)[:,1])) # 0.95042982815
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch2.predict_proba(X_test)[:,1])) # 0.8297293955


param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
# {'gamma': 0.2}, 0.82950732646119962
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch3.predict_proba(X_train)[:,1])) # 0.951596762733
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch3.predict_proba(X_test)[:,1])) # 0.829067732374

param_test3 = {'gamma':[i/100.0 for i in range(18,35)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
# {'gamma': 0.2}, 0.82950732646119962
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch3.predict_proba(X_train)[:,1])) # 0.951596762733
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch3.predict_proba(X_test)[:,1])) # 0.829067732374



param_test4 = {'subsample':[i/10.0 for i in range(6,10)]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test4, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
# {'subsample': 0.8}, 0.82950732646119962
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch4.predict_proba(X_train)[:,1])) # 0.951596762733
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch4.predict_proba(X_test)[:,1])) # 0.829067732374


param_test5 = {'colsample_bytree':[i/10.0 for i in range(6,10)]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test5, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
# {'colsample_bytree': 0.7}, 0.83012312363434881
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch5.predict_proba(X_train)[:,1])) # 0.948674365459
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch5.predict_proba(X_test)[:,1])) # 0.829666969524

param_test6 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
param_test6 = {'reg_alpha':[0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=10, 
                                                  min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.7,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test6, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch6.fit(X_train,y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
# {'reg_alpha': 0.1}, 0.83021934854781132
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, gsearch6.predict_proba(X_train)[:,1])) # 0.947339819809
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, gsearch6.predict_proba(X_test)[:,1])) # 0.829605606926

# Lastly, we should lower the learning rate and add more trees.

xgbFull = XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=10,
                        min_child_weight=1, gamma=0.5, subsample=0.8, 
                        colsample_bytree=0.7, reg_alpha=0.001,
                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
xgbFull.fit(X_train,y_train,eval_metric='auc')
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train, xgbFull.predict_proba(X_train)[:,1]))
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test, xgbFull.predict_proba(X_test)[:,1]))

'''
XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.2,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.005,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993967481249
Cross-validation (AUC) on test data 0.8342783645

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.2,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.99430982923
Cross-validation (AUC) on test data 0.834163861493

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=15,min_child_weight=1,gamma=0.2,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.999988889765
Cross-validation (AUC) on test data 0.831808422608                 
           
XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=5,min_child_weight=1,gamma=0.2,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.909030417641
Cross-validation (AUC) on test data 0.828334258013 
                 
XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.9,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.992107458315
Cross-validation (AUC) on test data 0.834170312969                 

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=2,gamma=0.2,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.005,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)                 
Cross-validation (AUC) on train data 0.98820867508
Cross-validation (AUC) on test data 0.832701849883                 

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.25,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993831413802
Cross-validation (AUC) on test data 0.834534510385                 

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.6,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.005,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993185153813
Cross-validation (AUC) on test data 0.834235791409

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.4,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.005,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993693300072
Cross-validation (AUC) on test data 0.834313679088

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.01,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993506309018
Cross-validation (AUC) on test data 0.83426802505

XGBClassifier(learning_rate=0.01,n_estimators=5000,max_depth=10,min_child_weight=1,gamma=0.5,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.005,objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=27)
Cross-validation (AUC) on train data 0.993483151359
Cross-validation (AUC) on test data 0.834643003408 

                 




      
'''              
### On null hearing date dataset


xgbNull = XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=10,
                        min_child_weight=1, gamma=0.2, subsample=0.8, 
                        colsample_bytree=0.7, reg_alpha=0.1,
                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
xgbNull.fit(X_train_sp,y_train_sp,eval_metric='auc')
print('Cross-validation (AUC) on train data', metrics.roc_auc_score(y_train_sp, xgbNull.predict_proba(X_train_sp)[:,1]))
print('Cross-validation (AUC) on test data', metrics.roc_auc_score(y_test_sp, xgbNull.predict_proba(X_test_sp)[:,1]))


##################################################################################################################################################################

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
testFeeNotWaived1['time_to_hearing'] = testFeeNotWaived1['time_to_hearing']/ np.timedelta64(1, 's')

testFeeNotWaived1 = testFeeNotWaived1.drop('index', axis=1)

testHasHearingDate = testFeeNotWaived1[~pd.isnull(testFeeNotWaived1['hearing_date'])]
testDNHHearingDate = testFeeNotWaived1[pd.isnull(testFeeNotWaived1['hearing_date'])]

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

     
      