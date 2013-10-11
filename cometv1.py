import numpy as np
import matplotlib.pyplot as pl
import emcee

def NormalAvgPrior(x, mu, sigma):
	pf = np.zeros(len(x))
	for i in range(len(x)):		
		logy = -np.log(sigma) - (x[i] - mu)**2 / (2.0*sigma**2)
		pf[i] = np.mean(np.exp(logy))
	return pf

def LogNormalAvgPrior(x, mu, sigma):
	pf = np.zeros(len(x))
	for i in range(len(x)):		
		logy = -np.log(sigma) - np.log(x[i]) - (np.log(x[i]) - mu)**2 / (2.0*sigma**2)
		pf[i] = np.mean(np.exp(logy))
	return pf

class cometClass:
	"""
	This class defines a comet with a certain total linear polarization and polarization angle. It also 
	contains the likelihood of the measured polarization states for many observers
	"""
	
	def __init__(self, StokesI, PL, angle, nObservers, noise):
		self.nObservers = nObservers
		self.I = StokesI
		self.PL = PL
		self.angle = angle		
		self.QPositive = np.zeros(nObservers) #np.random.uniform(0,np.pi,nObservers)
		self.noise = noise
		
		np.random.seed(seed=1234)
		
		self.modulation = np.zeros((nObservers,8))
		self.angleModulation = np.zeros((nObservers,8))
		for i in range(nObservers):
			QObs = self.PL * np.cos(2.0*(self.angle-self.QPositive[i]))
			UObs = self.PL * np.sin(2.0*(self.angle-self.QPositive[i]))
			self.angleModulation[i,:] = np.linspace(0,np.pi,8) - self.QPositive[i]
			self.modulation[i,:] = 0.5*(self.I + QObs*np.cos(2.0*self.angleModulation[i,:]) + UObs*np.sin(2.0*self.angleModulation[i,:])) + self.noise * np.random.randn(8)
	
	def logPosterior(self, x):
		logPL = x[0:self.nObservers]
		angle = x[self.nObservers:2*self.nObservers]
		QPositive = x[2*self.nObservers:3*self.nObservers]
		#StokesI = x[4*self.nObservers:5*self.nObservers]
		
		hyperPL = x[3*self.nObservers:3*self.nObservers+2]
						
		#if (np.any(StokesI < 0)):
			#return -np.inf
		
		if (hyperPL[1] < 0):
			return -np.inf
		
		#hyperAlpha = x[5*self.nObservers+2:5*self.nObservers+4]
		
# Data log-likelihood
		logL = 0.0
		
		QObs = 10**logPL * np.cos(2.0*(angle-self.QPositive))
		UObs = 10**logPL * np.sin(2.0*(angle-self.QPositive))
			
		for i in range(self.nObservers):			
			angleModulation = np.linspace(0,np.pi,8) - QPositive[i]
			model = 0.5*(1.0 + QObs[i]*np.cos(2.0*angleModulation) + UObs[i]*np.sin(2.0*angleModulation))
			logL += -np.sum((model-self.modulation[i,:])**2 / (2.0*self.noise**2))
			logL += -np.log(hyperPL[1]) - (logPL[i]-hyperPL[0])**2 / (2.0*hyperPL[1]**2)
				
		logL -= 1.0 / hyperPL[1]
		
		return logL
		
	def __call__(self, x):
		return self.logPosterior(x)
	

comet = cometClass(1.0,0.02, 0.35, 10, 0.005)
ndim = comet.nObservers * 3 + 2
nwalkers = 64

pl.close('all')
fig = pl.figure(num=0, figsize=(13,8))

#p0 = np.hstack([0.02*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),1.0*np.ones(comet.nObservers), [0.05,0.005]])
p0 = np.hstack([np.log10(0.02)*np.ones(comet.nObservers), 0.2*np.ones(comet.nObservers), 0.01*np.ones(comet.nObservers), [np.log10(0.01),0.01]])
p0 = emcee.utils.sample_ball(p0, np.ones(ndim)*0.001, size=nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, comet)
pos, prob, state = sampler.run_mcmc(p0, 200)
sampler.reset()
sampler.run_mcmc(pos, 200)

nRows = 3
nCols = 3
loop = 1
for i in range(comet.nObservers):
	ax = fig.add_subplot(nRows,nCols,loop)
	ax.plot(np.exp(sampler.flatchain[:,i]))
	loop += 1
	
x = np.linspace(-3,1,200)
alpha = sampler.flatchain[:,-2]
beta = sampler.flatchain[:,-1]
px = NormalAvgPrior(x, alpha, beta)
	
ax = fig.add_subplot(nRows,nCols,loop)
ax.plot(sampler.flatchain[:,-2])
loop += 1

ax = fig.add_subplot(nRows,nCols,loop)
ax.plot(sampler.flatchain[:,-1])
loop += 1

ax = fig.add_subplot(nRows,nCols,loop)
ax.plot(x, px)
ax.axvline(x=np.log10(0.02))

fig.tight_layout()