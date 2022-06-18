import pandas as pd
import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
import swifter
from model import *

# model with incentives
df = pd.read_csv('data/estimation_sample.csv')
m = modelbase()
m.init_data(df,np.arange(1,8))
shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']

# estimates
pars = pd.read_excel('output/table_estimates_framing.xlsx')
itheta = pars['point'].to_numpy()
print(itheta)
m.init_params(shifters=shifters,iframe=True,itheta=itheta)
for p in m.params:
	print(p.label, p.value)
# compute alphas
m.compute_prob_alpha()

# compute effect of framing
table = pd.DataFrame(index=np.arange(60,70),columns=['prob','change','percent'],dtype='float64')

for a in np.arange(60,70):
	m.predict(claim_age=a,accrual_change=0.0,nra=65,scn='base')
	m.predict(claim_age=a,accrual_change=0.0,nra=67,scn='delay')
	table.loc[a,'prob'] = m.df['prob_base'].mean()
	table.loc[a,'delay'] = m.df['prob_delay'].mean()
	table.loc[a,'change'] = m.df['prob_delay'].mean()-m.df['prob_base'].mean()
	table.loc[a,'percent'] = m.df['prob_delay'].mean()/m.df['prob_base'].mean()-1

print(table.round(4))


table.round(4).to_latex('output/table_simulation_framing.tex')
table.round(3).to_excel('output/table_simulation_framing.xlsx')
print(table.sum())
