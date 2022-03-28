import pandas as pd 
import numpy as np 
from numba import njit, float64, int64
from numba.types import Tuple
import swifter 
from model import * 

# model with incentives
df = pd.read_csv('estimation_sample.csv')
m = modelbase()
m.init_data(df,np.arange(1,7))
shifters = ['female','college','married']
m.init_params(shifters=shifters)
m.estimate()
m.covar()
table = m.output()
table.to_latex('table_estimates_incentives_noshifters.tex')
table.to_excel('table_estimates_incentives_noshifters.xlsx')
print(table)
