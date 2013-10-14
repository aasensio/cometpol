import numpy as np
import matplotlib.pyplot as pl
import pymc as mc

# Generate artificial data
nObservers = 6
anglesModulation = [0.0, 45.0, 90.0, 135.0]
nAngles = len(anglesModulation)
PLOK = 0.02
alphaOK = 25.0
SIOK = 1.0
noise = 0.001
theta0OK = np.random.normal(loc=0.0,scale=0.1,size=nObservers)
modulation = np.zeros((nAngles,nObservers))
for i in range(nObservers):
	QObs = PLOK * np.cos(np.deg2rad(2.0*(alphaOK-theta0OK[i])))
	UObs = PLOK * np.sin(np.deg2rad(2.0*(alphaOK-theta0OK[i])))
	angles = np.deg2rad(anglesModulation - theta0OK[i])
	modulation[:,i] = 0.5*SIOK*(1.0 + QObs*np.cos(2.0*angles) + UObs*np.sin(2.0*angles)) + noise * np.random.randn(nAngles)

# Hyperpriors
tauTheta0 = mc.Uniform('sigmaTheta0', lower=0, upper=1.0/30.0**2)

# Priors
PL = mc.Uniform('PL', lower=0, upper=1.0)
alpha = mc.Uniform('alpha', lower=0, upper=180.0)
theta0 = mc.Normal('theta0_%i' % i, mu=np.zeros(nObservers), tau=np.ones(nObservers)*tauTheta0)
SI = mc.Normal('SI_%i' % i, mu=np.ones(nObservers), tau=np.ones(nObservers) * 1.0 / 0.2**2)

# Likelihood
obs = np.zeros((nAngles,nObservers))

for i in range(nObservers):
	for j in range(nAngles):
		@mc.deterministic(plot=False)
		def observer_i(PL=PL, alpha=alpha, theta0=theta0[i], SI=SI[i]):
			QI = PL * np.cos(np.deg2rad(2.0*(alpha-theta0)))
			UI = PL * np.sin(np.deg2rad(2.0*(alpha-theta0)))
			angles = np.deg2rad(anglesModulation[j] - theta0)
			return 0.5*SI*(1.0 + QI*np.cos(2.0*angles) + UI*np.sin(2.0*angles))
		obs[j,i] = mc.Normal('data_%i' % i, mu=observer_i, tau=1.0/noise**2, value=modulation[j,i], observed=True)
		

	