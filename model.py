from re import S
import numpy as np
import pandas as pd
from numba import njit, float64, int64
from scipy.optimize import minimize

class params:
	def __init__(self, label, ifree, value, se = np.nan):
		self.label = label
		self.ifree = ifree
		self.value = value
		self.se = se
		return


@njit
def F(V):
	return 1 - np.exp(-np.exp(V))

@njit
def value(t, nra, g, r, db, alpha, lambda_G, lambda_R, lambda_D, lambda_F):
	v = alpha + lambda_G*g
	if t>=nra:
		v += lambda_F[1]*(t-nra)
	if t<nra:
		v += lambda_F[0]*(nra-t)
	if t>=r:
		v += lambda_R
	if t>=db:
		v += lambda_D
	return v

@njit
def cprob_alpha_j(j, alpha, claim_age, nra, gs, r, db, lambda_G, lambda_R, lambda_D, lambda_F):
	prob = 1.0
	for t in range(60,70):
		V = value(t,nra,gs[t-60],r, db, alpha, lambda_G,lambda_R,lambda_D,lambda_F)
		if t<claim_age:
			prob *= 1-F(V)
		else :
			prob *= F(V)
			break
	return prob


@njit
def probi(claim_ages, nras, gs, r, db, X, lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis):
	J = claim_ages.shape[0]
	K = alphas.shape[0]
	S = X.shape[0]
	xb = 0.0
	for s in range(S):
		xb += lambda_X[s]*X[s]
	pi = 0.0
	for k in range(K):
		pik = 1.0
		for j in range(J):
			pik *= cprob_alpha_j(j, xb + alphas[k], claim_ages[j], nras[j],
					gs[j,:], r, db, lambda_G, lambda_R, lambda_D, lambda_F)
		pi += pis[k]*pik
	pi = max(pi,1e-10)
	return pi

@njit
def probi_alpha(k, claim_ages, nras, gs, r, db, X, lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis):
	J = claim_ages.shape[0]
	K = alphas.shape[0]
	S = X.shape[0]
	xb = 0.0
	for s in range(S):
		xb += lambda_X[s]*X[s]
	pik = 1.0
	for j in range(J):
		pik *= cprob_alpha_j(j, xb + alphas[k], claim_ages[j], nras[j],
				gs[j,:], r, db, lambda_G, lambda_R, lambda_D, lambda_F)
	return pik

@njit
def probi_sim(alpha, claim_age, nra, gs, r, db, X, lambda_X, lambda_G, lambda_R, lambda_D, lambda_F):
	S = X.shape[0]
	xb = 0.0
	for s in range(S):
		xb += lambda_X[s]*X[s]
	prob = 1.0
	for t in range(60,70):
		V = value(t,nra,gs[t-60],r, db, alpha, lambda_G, lambda_R,lambda_D,lambda_F)
		if t<claim_age:
			prob *= 1-F(V)
		else :
			prob *= F(V)
			break
	return prob

class modelbase:
	def __init__(self):
		self.ages = np.arange(60,71)
		self.nages = len(self.ages)
		pd.set_option('display.max_rows', 500)
		return
	def init_data(self,s_df, scenarios):
		df = s_df.loc[:,:]
		df.set_index('respid',inplace=True)
		old_labels = ['q37_'+str(j) for j in scenarios]
		claim_ages = ['claim_'+str(j) for j in scenarios]
		map_labels = dict(zip(old_labels,claim_ages))
		df = df.rename(map_labels,axis=1)
		keep = claim_ages[:]
		df['miss_retage'] = np.where(df['retage'].isna(),1.0,0.0)
		keep.append('miss_retage')
		df['retage'] = np.where(df['retage'].isna(),df['age'],df['retage'])
		keep.append('retage')
		df['female'] = df['female'].replace({'female':1,'male':0})
		keep.append('female')
		df['dbage'] = df.loc[:,'planclaimdb']
		df.loc[df['receivedb']==1.0,'dbage'] = df.loc[df['receivedb']==1.0,'age']
		df.loc[df['dbage'].isna(),'dbage'] = 85
		keep.append('dbage')
		df['college'] = df['educ_3cat'].replace({'Above HS':1,'HS':0,'less than HS':0})
		keep.append('college')
		df['pref_bequest'] = np.where((df['pref_bequest']=='Strongly agree') | (df['pref_bequest']=='Agree'),1,0)
		keep.append('pref_bequest')
		df['pref_raversion'] = np.where((df['pref_raversion']=='Below average risk') | (df['pref_raversion']=='No risk'),1,0)
		keep.append('pref_raversion')
		df['pref_livewell'] = np.where((df['pref_livewell']=='Strongly agree') | (df['pref_livewell']=='Agree'),1,0)
		keep.append('pref_livewell')
		df['pref_spendquickly'] = np.where((df['pref_spendquickly']=='Strongly agree') | (df['pref_spendquickly']=='Agree'),1,0)
		keep.append('pref_spendquickly')
		df['pref_patient'] = np.where((df['pref_patient']=='Strongly agree') | (df['pref_patient']=='Agree'),1,0)
		keep.append('pref_patient')
		df['log_wealth'] = np.log(1.0+df['wealth'])
		keep.append('log_wealth')
		df['log_earn'] = np.log(df['earn'])
		keep.append('log_earn')
		df['treat'] = df['treatment'].replace({'Control':0,'Insurance':1,'Break-even':2})
		keep.append('treat')
		for j in scenarios:
			for t in self.ages:
				keep.append('gaindelay_'+str(t)+'_'+str(j))
			df = df.loc[~df['gaindelay_'+str(60)+'_'+str(j)].isna(),:]
		self.gains_labels = ['gaindelay_'+str(a)+'_'+str(j) for j in scenarios for a in self.ages]
		for j in scenarios:
			df['nra_'+str(j)] = 65
			if j==7:
				df.loc[df.frame==2,'nra_'+str(j)] = 67
			keep.append('nra_'+str(j))
		keep.append('age')
		for v in ['married','healthproblems','finlit','retlit']:
			keep.append(v)
		df = df[keep]
		self.N = len(df)
		self.df = df
		self.J = len(scenarios)
		self.scenarios = scenarios
		print('* descriptive statistics: ', df.describe().transpose())
		print('- sample size and number of scenarios: ',self.N,self.J)
		return

	def init_params(self,shifters=['female'],  iaccrual=True, iret=True, idb = True, iframe=False, K=3, itheta=None):
		self.params = []
		# shifters
		self.shifters = shifters
		self.S = len(self.shifters)
		pos = 0
		for s in range(self.S):
			if itheta is not None:
				self.params.append(params(self.shifters[s],True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params(self.shifters[s],True,0.0))
		# incentives
		self.iaccrual = iaccrual
		if self.iaccrual:
			if itheta is not None:
				self.params.append(params('lambda_G',True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('lambda_G',True,0.0))
		else :
			self.params.append(params('lambda_G',False,0.0))
		# retirement
		self.iret = iret
		if self.iret:
			if itheta is not None:
				self.params.append(params('lambda_R',True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('lambda_R',True,0.0))
		else :
			self.params.append(params('lambda_R',False,0.0))
		# db
		self.idb = idb
		if self.idb:
			if itheta is not None:
				self.params.append(params('lambda_D',True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('lambda_D',True,0.0))
		else :
			self.params.append(params('lambda_D',False,0.0))
		# framing
		self.iframe = iframe
		if self.iframe:
			if itheta is not None:
				self.params.append(params('lambda_Fm',True,itheta[pos]))
				pos +=1
				self.params.append(params('lambda_Fp',True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('lambda_Fm',True,0.0))
				self.params.append(params('lambda_Fp',True,0.0))
		else :
			self.params.append(params('lambda_Fm',False,0.0))
			self.params.append(params('lambda_Fp',False,0.0))
			pos +=2
		# mass points
		self.K = K
		ivalues = [-2.0,-1.0,0.0,1.0,2.0]
		for k in range(self.K):
			if itheta is not None:
				self.params.append(params('alpha_'+str(k),True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('alpha_'+str(k),True,ivalues[k]))
		self.params.append(params('log_odd_pi_'+str(0),False,0.0))
		pos +=1
		for k in range(1,self.K):
			if itheta is not None:
				self.params.append(params('log_odd_pi_'+str(k),True,itheta[pos]))
				pos +=1
			else :
				self.params.append(params('log_odd_pi_'+str(k),True,0.0))


		# some accounting
		self.npar = len(self.params)
		self.nfreepars = 0
		for p in self.params:
			if p.ifree:
				self.nfreepars +=1
		return
	def extract_freepars(self):
		theta = []
		for p in self.params:
			if p.ifree:
				theta.append(p.value)
		return np.array(theta,dtype=np.float64)

	def set_freepars(self,theta):
		i = 0
		for p in self.params:
			if p.ifree:
				p.value = theta[i]
				i +=1
		return
	def parse_params(self):
		lambda_X = np.zeros(self.S)
		lambda_F = np.zeros(2)
		alphas = np.zeros(self.K)
		pis = np.zeros(self.K)
		s = 0
		k = 0
		i = 0
		for p in self.params:
			if p.label in self.shifters:
				lambda_X[s] = p.value
				s += 1
			if p.label=='lambda_G':
				lambda_G = p.value
			if p.label=='lambda_R':
				lambda_R = p.value
			if p.label=='lambda_D':
				lambda_D = p.value
			if p.label=='lambda_Fm':
				lambda_F[0] = p.value
			if p.label=='lambda_Fp':
				lambda_F[1] = p.value
			if p.label in ['alpha_'+str(k) for k in range(self.K)]:
				alphas[k] = p.value
				k +=1
			if p.label in ['log_odd_pi_'+str(k) for k in range(self.K)]:
				pis[i] = np.exp(p.value)
				i +=1
		pis = pis[:]/np.sum(pis)
		return lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis
	def probi_df(self, row):
		claim_ages = row[['claim_'+str(j) for j in self.scenarios]].astype('int64').to_numpy()
		ret_age = row['retage'].astype('int64')
		nras = row[['nra_'+str(j) for j in self.scenarios]].astype('int64').to_numpy()
		gs = row[self.gains_labels].astype('float64').to_numpy().reshape((self.J,11))
		X = row[self.shifters].astype('float64').to_numpy()
		db_age = row['dbage'].astype('int64')
		lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis = self.parse_params()
		return probi(claim_ages,nras,gs, ret_age, db_age, X,
			lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis)
	def loglike(self,theta):
		self.set_freepars(theta)
		probs = self.df.apply(self.probi_df,axis=1)
		return np.sum(np.log(probs))
	def loglike_i(self,theta):
		self.set_freepars(theta)
		probs = self.df.apply(self.probi_df,axis=1)
		return np.log(probs)
	def estimate(self):
		itheta = self.extract_freepars()
		neg_loglike = lambda theta: - self.loglike(theta)
		opt = minimize(neg_loglike,itheta,method='L-BFGS-B',options={'ftol':1e-6,'disp': True,'iprint':1} )
		opt_theta = opt.x
		print('* estimation completed...')
		self.set_freepars(opt_theta)
		self.opt_loglike = self.loglike(opt_theta)
		return
	def covar(self):
		theta = self.extract_freepars()
		grad = np.zeros((self.N,self.nfreepars),dtype=np.float64)
		fopt_i = self.loglike_i(theta)
		fup_i = np.zeros(self.N)
		eps = 1e-6
		for j in range(self.nfreepars):
			theta_up = theta[:]
			theta_up[j] += eps
			fup_i = self.loglike_i(theta_up)
			grad[:,j] = (fup_i - fopt_i)/eps

		B = grad.T @ grad
		self.covar = np.linalg.inv(B)
		ses = np.sqrt(np.diagonal(self.covar))
		i = 0
		for p in self.params:
			if p.ifree:
				p.se = ses[i]
				i +=1
		print('* computed standard errors ...')
		return
	def aic(self):
		return  2*self.nfreepars - 2*self.opt_loglike
	def output(self):
		labels = [p.label for p in self.params]
		point = [p.value for p in self.params]
		free = [p.ifree for p in self.params]
		ses = [p.se for p in self.params]
		estimates = pd.DataFrame(index=labels,columns=['point','se','free'])
		estimates['point'] = point
		estimates['se'] = ses
		estimates['free'] = free
		print('loglike = ', self.opt_loglike)
		print('aic = ',self.aic())
		return estimates

	def probi_alpha_df(self, row):
		claim_ages = row[['claim_'+str(j) for j in self.scenarios]].astype('int64').to_numpy()
		ret_age = row['retage'].astype('int64')
		nras = row[['nra_'+str(j) for j in self.scenarios]].astype('int64').to_numpy()
		gs = row[self.gains_labels].astype('float64').to_numpy().reshape((self.J,11))
		X = row[self.shifters].astype('float64').to_numpy()
		db_age = row['dbage'].astype('int64')
		lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis = self.parse_params()
		pi = probi(claim_ages,nras,gs, ret_age, db_age, X,
			lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis)
		for k in range(self.K):
			pik = probi_alpha(k,claim_ages,nras,gs, ret_age, db_age, X,
				lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis)
			row['pi_alpha_'+str(k)] = pik*pis[k]/pi
		pis_post = row[['pi_alpha_'+str(k) for k in range(self.K)]].to_numpy()
		newvars = ['pi_alpha_'+str(k) for k in range(self.K)]
		newvars.append('alpha_mean')
		newvars.append('alpha_mode')
		row['alpha_mean'] = np.sum([p*a for p,a in zip(pis_post,alphas)])
		row['alpha_mode'] = alphas[np.argmax(row[['pi_alpha_'+str(k) for k in range(self.K)]].to_numpy())]
		return row[newvars]
	def compute_prob_alpha(self):
		newvars = ['pi_alpha_'+str(k) for k in range(self.K)]
		newvars.append('alpha_mean')
		newvars.append('alpha_mode')
		self.df.loc[:,newvars] = np.nan
		self.df.loc[:,newvars] = self.df.apply(self.probi_alpha_df,axis=1)
		return
	def probi_sim_df(self,row, claim_age = 65, accrual_change=0.0, nra = 65):
		ret_age = row['retage'].astype('int64')
		gs = row[self.gains_labels].astype('float64').to_numpy().reshape((self.J,11))
		# always use actual incentives and apply change
		gs = gs[0,:] + accrual_change
		X = row[self.shifters].astype('float64').to_numpy()
		db_age = row['dbage'].astype('int64')
		lambda_X, lambda_G, lambda_R, lambda_D, lambda_F, alphas, pis = self.parse_params()
		alpha = row['alpha_mean'].astype('float64')
		pi = probi_sim(alpha, claim_age, nra, gs, ret_age, db_age, X, lambda_X, lambda_G, lambda_R, lambda_D, lambda_F)
		return pi
	def predict(self,claim_age,accrual_change=0.0, nra=65, scn='base'):
		self.df['prob_'+scn] = self.df.apply(self.probi_sim_df,axis=1,args=(claim_age, accrual_change, nra,))
		return










