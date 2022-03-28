import pandas as pd 
import numpy as np 
from numba import njit, float64, int64
from numba.types import Tuple
import swifter 
from model import * 

# model for each treatment group
for v in ['Control','Insurance','Break-even']:
	df = pd.read_csv('estimation_sample.csv')
	df = df.loc[df['treatment']==v,:]
	m = modelbase()
	m.init_data(df,np.arange(2,8))
	shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']
	m.init_params(shifters=shifters,iframe=True)
	m.estimate()
	m.covar()
	table = m.output()
	table.to_latex('table_estimates_education_'+v+'.tex')
	table.to_excel('table_estimates_education_'+v+'.xlsx')
	print('Arm = ', v)
	print(table)
