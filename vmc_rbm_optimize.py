import numpy as np 
import cmath
import numba
import time
from matplotlib import pyplot


@numba.jit
def log_derivative(state, alpha, Nsite):

	mat = np.zeros((Nsite, Nsite), dtype = np.cdouble)

	mvpd = np.zeros(Nsite, dtype = np.cdouble)
	for i in range(Nsite):

		ssum = 0.0
		for j in range(Nsite):
			ssum += alpha[i][j] * state[j]

		mvpd[i] = ssum; 


	mat = np.outer( np.tanh(mvpd), state) 

	return mat


@numba.jit
def coefficient(state, alpha, Nsite):

	res = 1.0
	mvpd = np.zeros(Nsite, dtype = np.cdouble)

	# use for loop instead of np.prod to use numba
	for i in range(Nsite):

		ssum = 0.0
		for j in range(Nsite):
			ssum += alpha[i][j] * state[j]

		mvpd[i] = ssum; 

	return np.prod(np.cosh(mvpd))


@numba.jit
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

	return res + 0.5 * ssum


@numba.jit
def metropolis(alpha, Nsite, Nsample=2000, Nskip = 3):

	state = np.ones(Nsite) 
	# state = state.astype(np.double)
	state[: Nsite//2] = -1

	state *= 0.5
	state = state[np.random.permutation(Nsite)]

	energy_sum = 0.0
	logder_sum = np.zeros((Nsite,Nsite), dtype = np.cdouble)
	HO_ssum = np.zeros((Nsite,Nsite), dtype = np.cdouble)

	flat_logder_sum = np.zeros(Nsite*Nsite, dtype = np.cdouble)
	logder_outer_sum = np.zeros((Nsite*Nsite, Nsite*Nsite), dtype = np.cdouble)



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

			if(np.random.random() < min(1.0, np.abs(coeff_new/coeff_old) ) ):
				state = new_state.copy()
				coeff_old = coeff_new

		
		tmp_energy = local_energy(state, coeff_old, alpha, Nsite)
		tmp_logder = log_derivative(state, alpha, Nsite)

		# natural gradient descent 
		flat_logder_sum += tmp_logder.flatten()
		logder_outer_sum += np.outer( np.conjugate(tmp_logder.flatten()),  tmp_logder.flatten() )

		tmp_logder = np.conjugate(tmp_logder)		

		energy_sum += tmp_energy
		logder_sum += tmp_logder
		HO_ssum += tmp_logder * tmp_energy


	energy_sum /= Nsample
	logder_sum /= Nsample
	HO_ssum /= Nsample

	flat_logder_sum /= Nsample
	logder_outer_sum /= Nsample


	logder_outer_sum -= np.outer( np.conjugate(flat_logder_sum), flat_logder_sum)


	gradient = HO_ssum  - logder_sum * energy_sum

	gradient_para = gradient.flatten() 
	logder_outer_sum += np.identity(Nsite*Nsite) * 1e-5

	derivative = np.linalg.solve(logder_outer_sum, gradient_para)

	derivative = derivative.reshape((Nsite, Nsite))


	return  energy_sum, derivative




def optimize(alpha, Nsite, Nsample, lamda):

	s = []
	y_energy = []
	t0 = time.time()

	# fp = open("energy_rbm_ngd.txt", "w")

	for i in range(50):
		energy, gradient = metropolis(alpha, Nsite, Nsample)
		# derivative = 2*hosum - 2 * logder * energy
		# print(alpha, energy, derivative)
		print("Step %d, energy %.4f\n" %(i, energy.real))
		# fp.write("%d  %.4f\n" %(i, energy.real))
		# print(hosum, ohsum, partial, energy)
		s.append(i)
		y_energy.append(energy)
		alpha = alpha - lamda * gradient

	# fp.close()
	t1 = time.time()
	print("Elapsed time: %.2f sec" % (t1 - t0))

	pyplot.plot(s, np.real(np.array(y_energy)) )
	pyplot.xlabel("Step")
	# pyplot.legend()
	pyplot.ylabel("Energy")
	pyplot.title("Variational energy of RBM wave function")
	pyplot.show()


if(__name__ == '__main__'):

	Nsite = 8
	Nsample = 5000
	lamda = 0.2
	re = np.random.random((Nsite, Nsite))
	im = np.random.random((Nsite, Nsite))
	alpha = re + 1j*im

	optimize(alpha, Nsite, Nsample, lamda)



