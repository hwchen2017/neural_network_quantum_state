import numpy as np 
import math
import numba
import time
from matplotlib import pyplot


@numba.jit(nopython = True)
def log_derivative(state, alpha, Nsite):

	ssum = 0.0
	for i in range(Nsite):
		for j in range(i+1, Nsite):
			deno = min(math.fabs(1.0*j - 1.0*i), Nsite*1.0 - math.fabs(1.0*j - 1.0*i))
			ssum += state[i] * state[j] / deno

	return ssum

@numba.jit(nopython = True)
def coefficient(state, alpha, Nsite):

	ssum = log_derivative(state, alpha, Nsite)
	return math.exp(-alpha * ssum) 

@numba.jit(nopython = True)
def local_energy(state, coeff, alpha, Nsite):

	res = 0.0

	for i in range(Nsite):
		res += state[i] * state[(i+1)%Nsite]

	ssum = 0.0

	for i in range(Nsite):
		if(state[i] * state[(i+1)%Nsite] < 0.0):
			
			state_new = state.copy()
			state_new[i] *= -1.0
			state_new[(i+1)%Nsite] *= -1.0

			ssum += coefficient(state_new, alpha, Nsite)/coeff

	return res - 0.5 * ssum


@numba.jit(nopython = True)
def metropolis(alpha, Nsite, Nsample=2000, Nskip = 3):

	state = np.ones(Nsite) 
	state[: Nsite//2] = -1

	state *= 0.5
	state = state[np.random.permutation(Nsite)]

	energy_sum = 0.0
	logder_sum = 0.0
	HO_ssum = 0.0


	for i in range(Nsample):

		for j in range(Nskip):
			x = np.random.randint(low = 0, high = Nsite)
			y = x

			while(state[y] * state[x] > 0):
				y = np.random.randint(low = 0, high = Nsite)

			new_state = state.copy()
			new_state[x] *= -1.0
			new_state[y] *= -1.0

			coeff_old = coefficient(state, alpha, Nsite)
			coeff_new =	coefficient(new_state, alpha, Nsite)

			if(np.random.random() < min(1.0, (coeff_new**2)/(coeff_old**2))):
				state = new_state.copy()
				coeff_old = coeff_new

		
		tmp_energy = local_energy(state, coeff_old, alpha, Nsite)
		tmp_logder = -1.0*log_derivative(state, alpha, Nsite)
		# tmp_oh = OH_energy(new_state, coeff_old, alpha)
		tmp_ho = tmp_energy * tmp_logder

		energy_sum += tmp_energy
		logder_sum += tmp_logder
		HO_ssum += tmp_ho

	return  HO_ssum/Nsample, logder_sum/Nsample, energy_sum / Nsample 



def optimize(alpha, Nsite, Nsample, lamda):

	x = []
	s = []
	y_energy = []
	d = []

	t0 = time.time()
	for i in range(100):
		hosum, logder, energy = metropolis(alpha, Nsite, Nsample)
		derivative = 2*hosum - 2 * logder * energy
		x.append(alpha)
		s.append(i)
		d.append(derivative)
		y_energy.append(energy)
		# print("%.3f %.6f %.6f\n" %(alpha, energy, derivative))
		# print(alpha, energy, derivative)

		alpha = alpha - lamda * derivative

	t1 = time.time()
	print("Elapsed time: %.2f sec" % (t1-t0))


	fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3)
	ax1.plot(s, y_energy, color = "darkred", label = "Variational Energy")
	ax1.plot(s, y_energy, "P", markersize = 3)
	ax2.plot(s, x, color = "navy", label = "Parameter: Alpha")

	ax3.plot(s, d, color = "purple", label = "Gradient")
	

	ax1.set(xlabel = "Step", ylabel = "Energy")
	ax2.set(xlabel = "Step")
	ax3.set(xlabel = "Step")
	# pyplot.xlabel("Step")
	ax1.legend()
	ax2.legend()
	ax3.legend()

	pyplot.suptitle("Optimizing single parameter Jastrow wave function")
	pyplot.show()


if(__name__ == '__main__'):

	Nsite = 4
	Nsample = 5000
	lamda = 0.2
	alpha = -3.0

	optimize(alpha, Nsite, Nsample, lamda)



