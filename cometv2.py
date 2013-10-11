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

def sigmoid(x, L, U):
	return L + (U-L) / (1.0+np.exp(-x))

def invSigmoid(x, L, U):
	return np.log((L-x)/(x-U))

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
		self.QPositive = np.random.normal(loc=0.0,scale=0.1,size=nObservers)
		self.noise = noise
		self.deltaT = 1.0
		self.anglesLow = -np.pi/2
		self.anglesUp = np.pi/2
		
		np.random.seed(seed=1234)
		
		self.modulation = np.zeros((nObservers,8))
		self.angleModulation = np.zeros((nObservers,8))
		
# Generate the observations
		for i in range(nObservers):
			QObs = self.PL * np.cos(2.0*(self.angle-self.QPositive[i]))
			UObs = self.PL * np.sin(2.0*(self.angle-self.QPositive[i]))
			self.angleModulation[i,:] = np.linspace(0,np.pi,8) - self.QPositive[i]
			self.modulation[i,:] = 0.5*(self.I + QObs*np.cos(2.0*self.angleModulation[i,:]) + UObs*np.sin(2.0*self.angleModulation[i,:])) + np.linspace(0,7*self.deltaT,8) + self.noise * np.random.randn(8)
	
#----------------------------
# Log-posterior
#----------------------------
	def logPosterior(self, x):
		PL = sigmoid(x[0], 1e-5, 1.0)
		angle = sigmoid(x[1], self.anglesLow, self.anglesUp)
		QPositive = sigmoid(x[2:2+self.nObservers], self.anglesLow, self.anglesUp)
		QPositiveMu = sigmoid(x[-3], 2*self.anglesLow, 2*self.anglesUp)
		QPositiveSigma = np.exp(x[-2])
		SI = np.exp(x[-1])
									
# Data log-likelihood
		logL = 0.0
				
		QObs = PL * np.cos(2.0*(angle-QPositive))		
		UObs = PL * np.sin(2.0*(angle-QPositive))
			
		for i in range(self.nObservers):			
			angleModulation = np.linspace(0,np.pi,8) - QPositive[i]			
			model = 0.5*(SI + QObs[i]*np.cos(2.0*angleModulation) + UObs[i]*np.sin(2.0*angleModulation)) + np.linspace(0,7*self.deltaT,8)
			logL += -np.sum((model-self.modulation[i,:])**2 / (2.0*self.noise**2))
			
# Prior for the theta0 reference angle
		logL += -self.nObservers * np.log(QPositiveSigma) - np.sum((QPositive - QPositiveMu)**2 / (2.0*QPositiveSigma**2))
		
		logL += -((SI - 1.0)**2 / (2.0*0.2**2))
		
		#logL += 1.0 / QPositiveSigma
						
		return logL
		
	def __call__(self, x):
		return self.logPosterior(x)
	

comet = cometClass(1.0,0.02, 0.35, 6, 0.01)
ndim = comet.nObservers + 5
nwalkers = comet.nObservers * 5

pl.close('all')
fig = pl.figure(num=0, figsize=(10,12))

PLInitial = invSigmoid(0.02,1e-5,1.0)
angleInitial = invSigmoid(0.35,comet.anglesLow, comet.anglesUp)
QPositiveInitial = invSigmoid(0.0,comet.anglesLow, comet.anglesUp)
QPositiveMuInitial = invSigmoid(0.0,2*comet.anglesLow, 2*comet.anglesUp)
QPositiveSigmaInitial = np.log(0.1)
SIInitial = np.log(1.0)

p0 = np.hstack([PLInitial, angleInitial, QPositiveInitial*np.ones(comet.nObservers), QPositiveMuInitial, QPositiveSigmaInitial, SIInitial])
stdDev = np.hstack([0.01, 0.01, 0.01*np.ones(comet.nObservers), 0.01, 0.01, 0.1])
p0 = emcee.utils.sample_ball(p0, stdDev, size=nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, comet)
pos, prob, state = sampler.run_mcmc(p0, 50)
sampler.reset()
sampler.run_mcmc(pos, 200)

sampler.flatchain[:,0] = sigmoid(sampler.flatchain[:,0],1e-5,1.0)
sampler.flatchain[:,1] = sigmoid(sampler.flatchain[:,1],comet.anglesLow, comet.anglesUp)
sampler.flatchain[:,-3] = sigmoid(sampler.flatchain[:,-3],2*comet.anglesLow, 2*comet.anglesUp)
sampler.flatchain[:,-2] = np.log10(np.exp(sampler.flatchain[:,-2]))
sampler.flatchain[:,-1] = np.exp(sampler.flatchain[:,-1])
loop = 1
valuesOk = [0.02,0.35, 0.0, 0.1, 1.0]
whichPar = [0,1,-3,-2,-1]
for i in range(len(whichPar)):
	ax = fig.add_subplot(5,2,loop)
	ax.plot(sampler.flatchain[:,whichPar[i]])
	loop += 1
	
	ax = fig.add_subplot(5,2,loop)
	ax.hist(sampler.flatchain[:,whichPar[i]], bins=20, histtype='step')
	ax.axvline(x=valuesOk[i])
	loop += 1

fig.tight_layout()