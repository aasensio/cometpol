import numpy as np
import matplotlib.pyplot as pl
import emcee

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
		self.angle0 = np.zeros(nObservers) #np.random.uniform(0,np.pi,nObservers)
		self.QPositive = np.zeros(nObservers) #np.random.uniform(0,np.pi,nObservers)
		self.noise = noise
		
		self.modulation = np.zeros((nObservers,8))
		self.angleModulation = np.zeros((nObservers,8))
		for i in range(nObservers):
			QObs = self.PL * np.cos(2.0*(self.angle-self.angle0[i]))
			UObs = self.PL * np.sin(2.0*(self.angle-self.angle0[i]))
			self.angleModulation[i,:] = np.linspace(0,np.pi,8) + self.QPositive[i]
			self.modulation[i,:] = 0.5*(self.I + QObs*np.cos(2.0*self.angleModulation[i,:]) + UObs*np.sin(2.0*self.angleModulation[i,:])) + self.noise * np.random.randn(8)
	
	def logPosterior(self, x):
		PL = x[0:self.nObservers]
		#angle = x[self.nObservers:2*self.nObservers]
		#angle0 = x[2*self.nObservers:3*self.nObservers]
		#QPositive = x[3*self.nObservers:4*self.nObservers]
		#StokesI = x[4*self.nObservers:5*self.nObservers]
		
		hyperPL = x[1*self.nObservers:1*self.nObservers+2]
		
		if (np.any(PL < 0)):
			return -np.inf
		
		#if (np.any(StokesI < 0)):
			#return -np.inf
		
		if (np.any(hyperPL < 0)):
			return -np.inf
		
		#hyperAlpha = x[5*self.nObservers+2:5*self.nObservers+4]
		
# Data log-likelihood
		logL = 0.0
		for i in range(self.nObservers):
			QObs = PL[i] * np.cos(2.0*(self.angle-self.angle0[i]))
			UObs = PL[i] * np.sin(2.0*(self.angle-self.angle0[i]))
			angleModulation = np.linspace(0,np.pi,8) + self.QPositive[i]#QPositive[i]			
			model = 0.5*(1.0 + QObs*np.cos(2.0*angleModulation) + UObs*np.sin(2.0*angleModulation))
			logL -= np.sum((model-self.modulation[i,:])**2 / (2.0*self.noise**2))
		
# Prior for PL
		logL += -self.nObservers * np.log(hyperPL[1]) - np.sum((PL-hyperPL[0])**2 / (2.0*hyperPL[1]**2))
		
		#logL -= 1.0 / hyperPL[1]
		
		return logL
		
	def __call__(self, x):
		return self.logPosterior(x)
	
	
comet = cometClass(1.0,0.01, 0.35, 5, 0.0001)
ndim = comet.nObservers * 1 + 2
nwalkers = 2*ndim

#p0 = np.hstack([0.02*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),0.2*np.ones(comet.nObservers),1.0*np.ones(comet.nObservers), [0.05,0.005]])
p0 = np.hstack([0.02*np.ones(comet.nObservers), [0.01,0.005]])
p0 = emcee.utils.sample_ball(p0, np.ones(ndim)*0.001, size=nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, comet)
pos, prob, state = sampler.run_mcmc(p0, 200)
sampler.reset()
sampler.run_mcmc(pos, 200)

pl.close('all')
fig = pl.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(sampler.flatchain[:,-2])
ax = fig.add_subplot(1,2,2)
ax.plot(sampler.flatchain[:,-1])