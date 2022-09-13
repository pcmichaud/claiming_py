import pandas as pd
import numpy as np 
from numba import njit, float64, int64
from numba.types import Tuple
import swifter
from model import *

# model with incentives
df = pd.read_csv('data/estimation_sample.csv')
m = modelbase()
m.init_data(df,np.arange(1,7))
shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']
m.init_params(shifters=shifters)
m.estimate()
m.covar()
table = m.output()
table.to_latex('output/table_estimates_incentives_lt.tex')
table.to_excel('output/table_estimates_incentives_lt.xlsx')
print(table)
