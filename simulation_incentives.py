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
shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']

# estimates
pars = pd.read_excel('table_estimates_framing.xlsx')
itheta = pars['point'].to_numpy()
print(itheta)
m.init_params(shifters=shifters,iframe=True,itheta=itheta)

for p in m.params:
	print(p.label, p.value)
# compute alphas 
m.compute_prob_alpha()

print(m.df[['pi_alpha_'+str(k) for k in range(m.K)]].describe())

for v in m.shifters:
	print(v)
	print(m.df.groupby(v).mean()[['alpha_mean','alpha_mode']])

print(m.df[['alpha_mean','alpha_mode']].describe())

# compute effect of incentives 
table = pd.DataFrame(index=np.arange(60,70),columns=['prob','accrual','change prob','elasticity'],dtype='float64')
daccrual = 2.5
for a in np.arange(60,70):
	m.predict(claim_age=a,accrual_change=0.0,nra=65,scn='base')
	m.predict(claim_age=a,accrual_change=daccrual,nra=65,scn='delay')
	mean_accrual = m.df['gaindelay_'+str(a)+'_1'].mean()
	table.loc[a,'prob'] = m.df['prob_base'].mean()
	table.loc[a,'accrual'] = mean_accrual
	table.loc[a,'change prob'] = m.df['prob_delay'].mean()-m.df['prob_base'].mean()
	table.loc[a,'elasticity'] = (m.df['prob_delay'].mean()/m.df['prob_base'].mean()-1)/(daccrual/mean_accrual)
print(table.round(4))

table.round(4).to_latex('table_simulation_incentives.tex')


print(table.sum())
