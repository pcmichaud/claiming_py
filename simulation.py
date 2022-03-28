import pandas as pd 
import numpy as np 
from numba import njit, float64, int64
from numba.types import Tuple
import swifter 
from model import * 

# # model with incentives
# df = pd.read_csv('estimation_sample.csv')
# m = modelbase()
# m.init_data(df,np.arange(1,7))
# shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']

# # estimates
# pars = pd.read_excel('table_estimates_framing.xlsx')
# itheta = pars['point'].to_numpy()
# print(itheta)
# m.init_params(shifters=shifters,iframe=True,itheta=itheta)

# for p in m.params:
# 	print(p.label, p.value)
# # compute alphas 
# m.compute_prob_alpha()

# print(m.df[['pi_alpha_'+str(k) for k in range(m.K)]].describe())

# for v in m.shifters:
# 	print(v)
# 	print(m.df.groupby(v).mean()[['alpha_mean','alpha_mode']])

# print(m.df[['alpha_mean','alpha_mode']].describe())


# # compute effect of incentives 
# daccrual = 1.0
# for a in np.arange(60,70):
# 	m.predict(claim_age=a,accrual_change=0.0,nra=65,scn='base')
# 	m.predict(claim_age=a,accrual_change=daccrual,nra=65,scn='delay')
# 	mean_accrual = m.df['gaindelay_'+str(a)+'_1'].mean()
# 	print('age = ',a,(m.df['prob_delay'].mean()/m.df['prob_base'].mean()-1)/(daccrual/mean_accrual),m.df['prob_delay'].mean()-m.df['prob_base'].mean(),mean_accrual)

# # compute effect of framing 
# for a in np.arange(60,70):
# 	m.predict(claim_age=a,accrual_change=0.0,nra=65,scn='base')
# 	m.predict(claim_age=a,accrual_change=0.0,nra=67,scn='delay')
# 	print('age = ',a,m.df['prob_base'].mean(),(m.df['prob_delay'].mean()/m.df['prob_base'].mean()-1),m.df['prob_delay'].mean()-m.df['prob_base'].mean())

# compute effect of education 

# first get the right alphas for each respondent, given their treatment group. Assume those are invariant no matter what happens
dfs = []
for v in ['Control','Break-even','Insurance']:
	df = pd.read_csv('estimation_sample.csv')
	df = df.loc[df['treatment']==v,:]
	m = modelbase()
	m.init_data(df,np.arange(2,7))
	shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']
	pars = pd.read_excel('table_estimates_education_'+v+'.xlsx')
	itheta = pars['point'].to_numpy()
	m.init_params(shifters=shifters,iframe=True,itheta=itheta)
	# compute alphas 
	m.compute_prob_alpha()
	dfs.append(m.df['alpha_mean'])
alphas = pd.concat(dfs,axis=0)

# estimates
table = pd.DataFrame(index=np.arange(60,70),columns=['Control','Break-Even','Insurance'])

for v in ['Control','Break-Even','Insurance']:
	df = pd.read_csv('estimation_sample.csv')
	m = modelbase()
	m.init_data(df,np.arange(2,7))
	shifters = ['female','college','married','healthproblems','finlit','retlit','pref_bequest','pref_patient','log_earn','log_wealth']
	pars = pd.read_excel('table_estimates_education_'+v+'.xlsx')
	itheta = pars['point'].to_numpy()
	print(itheta)
	m.init_params(shifters=shifters,iframe=True,itheta=itheta)
	for p in m.params:
		print(p.label, p.value)
	print(v)
	m.df.loc[:,'alpha_mean'] = alphas
	for a in np.arange(60,70):
		m.predict(claim_age=a,accrual_change=0.0,nra=65,scn='base')
		print(a,m.df['prob_base'].mean())
		table.loc[a,v] = m.df['prob_base'].mean()

print(table)
