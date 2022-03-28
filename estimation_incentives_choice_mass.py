import pandas as pd
import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
import swifter
from model import *

# model with incentives
Ks = [2,3,4,5]
aics = []
for k in Ks:
	df = pd.read_csv('data/estimation_sample.csv')
	m = modelbase()
	m.init_data(df,np.arange(1,7))
	shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']
	m.init_params(shifters=shifters,K=k)
	m.estimate()
	m.covar()
	table = m.output()
	aics.append(m.aic())
	table.to_latex('output/table_estimates_incentives_mass_'+str(k)+'.tex')
	table.to_excel('output/table_estimates_incentives_mass_'+str(k)+'.xlsx')
	print('K = ',k)
	print(table)
results = pd.DataFrame(index=Ks,columns=['aic'])
results['aic'] = aics
print(results)
results.to_excel('output/table_estimates_aic_pick_K.xlsx')
