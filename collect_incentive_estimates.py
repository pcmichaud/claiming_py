import pandas as pd
import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
from model import *

# model for each treatment group
estimates = []
for v in ['subjective','compas','lt']:
	df = pd.read_excel('output/table_estimates_incentives_'+v+'.xlsx')
	df.columns = ['param','point','se','free']
	df = df[df['free']]
	df.drop('free',axis=1,inplace=True)
	df.set_index('param',inplace=True)
	#print(df.stack())
	estimates.append(df.stack())

table = pd.concat(estimates,axis=1)
table.columns = ['Subjective','Compas','Life-Table']

table.loc['loglike',:] = np.nan
print(table)
table.to_latex('output/table_estimates_incentives_joint.tex')

