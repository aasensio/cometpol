import numpy as np
import matplotlib.pyplot as pl
import pymc as mc

# Generate artificial data
nObservers = 6
anglesModulation = np.asarray([0.0, 45.0, 90.0, 135.0])
nAngles = len(anglesModulation)
PLOK = 0.02
alphaOK = 25.0
SIOK = 1.0
noise = 0.01
theta0OK = np.random.normal(loc=0.0,scale=10.0,size=nObservers)
modulation = np.zeros((nAngles,nObservers))

QObs = PLOK * np.cos(np.deg2rad(2.0*(alphaOK-theta0OK)))
QObs = QObs[:,np.newaxis]
UObs = PLOK * np.sin(np.deg2rad(2.0*(alphaOK-theta0OK)))
UObs = UObs[:,np.newaxis]

angles = np.deg2rad(anglesModulation[np.newaxis,:] - theta0OK[:,np.newaxis])
modulation = 0.5*SIOK*(1.0 + QObs*np.cos(2.0*angles) + UObs*np.sin(2.0*angles)) + noise * np.random.randn(nObservers,nAngles)

# Hyperpriors
sigmaTheta0 = mc.Uniform('sigmaTheta0', lower=0, upper=40.0)

@mc.deterministic(plot=False)
def precision(sigmaTheta0=sigmaTheta0):
	return 1.0 / sigmaTheta0**2

## Priors
PL = mc.Uniform('PL', lower=0, upper=1.0)
alpha = mc.Uniform('alpha', lower=0, upper=180.0)
theta0 = mc.Normal('theta0', mu=np.zeros(nObservers), tau=np.ones(nObservers)*precision)
SI = mc.Normal('SI', mu=np.ones(nObservers), tau=np.ones(nObservers) * 1.0 / 0.2**2)
 
@mc.deterministic(plot=False)
def observer_i(PL=PL, alpha=alpha, theta0=theta0, SI=SI):
	QI = PL * np.cos(np.deg2rad(2.0*(alpha-theta0)))
	QI = QI[:,np.newaxis]
	UI = PL * np.sin(np.deg2rad(2.0*(alpha-theta0)))
	UI = UI[:,np.newaxis]
	
	SI = SI[:,np.newaxis]

	angles = np.deg2rad(anglesModulation[np.newaxis,:] - theta0[:,np.newaxis])
	return 0.5*SI*(1.0 + QI*np.cos(2.0*angles) + UI*np.sin(2.0*angles))

## Likelihood
#obs = np.zeros((nAngles,nObservers))

#for i in range(nObservers):
	#for j in range(nAngles):
		
obs = mc.Normal('data', mu=observer_i, tau=1.0/noise**2, value=modulation, observed=True)	

model = mc.Model([PL, alpha, theta0, SI, obs]) 
mcmc = mc.MCMC(model)
mcmc.sample(iter=10000,burn=5000, thin=2)

var = ['PL','alpha']
fig = pl.figure(num=0, figsize=(12,8))
loop = 1
for i in range(2):
	res = mcmc.trace(var[i]).gettrace()
	ax = fig.add_subplot(2,2,loop)
	ax.plot(res)
	loop += 1
	ax = fig.add_subplot(2,2,loop)
	ax.hist(res)
	loop += 1