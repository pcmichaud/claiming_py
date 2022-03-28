import pandas as pd 
import numpy as np 
from numba import njit, float64, int64
from numba.types import Tuple
import swifter 
from model import * 

# model for each treatment group
estimates = []
for v in ['Control','Insurance','Break-even']:
	df = pd.read_excel('table_estimates_education_'+v+'.xlsx')
	df.columns = ['param','point','se','free']
	df = df[df['free']]
	df.drop('free',axis=1,inplace=True)
	df.set_index('param',inplace=True)
	#print(df.stack())
	estimates.append(df.stack())

table = pd.concat(estimates,axis=1)
table.columns = ['Control','Insurance','Break-even']

table.loc['loglike',:] = np.nan
print(table)
table.to_latex('table_estimates_education_joint.tex')