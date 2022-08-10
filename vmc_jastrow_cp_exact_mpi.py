import numpy as np
from mpi4py import MPI
import math
import time
import numba
from matplotlib import pyplot

N = 4

@numba.njit
def coefficient(state, alpha):

	ssum = 0.0
	for i in range(N):
		for j in range(i+1, N):

			deno = min(math.fabs(1.0*j - 1.0*i), N*1.0 - math.fabs(1.0*j - 1.0*i) )
			# print(state[i] * state[j] / deno)
			ssum += state[i] * state[j] / deno

	return math.exp(-alpha * ssum) 

@numba.njit
def local_energy(state, coeff, alpha):

	res = 0.0
	ssum = 0.0 

	for i in range(N):
		res += state[i] * state[(i+1)%N]

	for i in range(N):
		if(state[i] * state[(i+1)%N] < 0.0):			
			state_new = state.copy()
			# print(state_new)
			state_new[i] *= -1.0
			state_new[(i+1)%N] *= -1.0

			ssum += coefficient(state_new, alpha)/coeff

	return res - 0.5 * ssum


@numba.njit
def sampler(alpha, Nsample = 5000, Nskip = 3):

	state = np.ones(N) 
	state[: N//2] = -1

	state *= 0.5
	state = state[np.random.permutation(N)]

	ssum = 0.0
	# coeff_old = coefficient(state, alpha)

	for i in range(Nsample):

		for i in range(Nskip):

			x = np.random.randint(low = 0, high = N)
			y = x

			while(state[y] * state[x] > 0):
				y = np.random.randint(low = 0, high = N)

			new_state = state.copy()
			new_state[x] *= -1.0
			new_state[y] *= -1.0

			coeff_old = coefficient(state, alpha)
			coeff_new =	coefficient(new_state, alpha)

			if(np.random.random() < min(1.0, (coeff_new**2)/(coeff_old**2))):
				state = new_state.copy()
				coeff_old = coeff_new

		tmp = local_energy(state, coeff_old, alpha)
		

		ssum += tmp

	return ssum / Nsample 


if(__name__ == '__main__'):

	comm = MPI.COMM_WORLD
	nprocs = comm.Get_size()
	rank = comm.Get_rank()

	ns = 10000
	ns = ns // nprocs


	if(rank == 0):
		x, y = [], []


	t0 = time.time()

	for i in range(-30, 40):

		alpha = i * 0.1

		# comm.Barrier()
		mpi_energy = sampler(alpha, ns) / nprocs

		energy = comm.reduce(mpi_energy, root=0)

		if(rank == 0):
			print("Alpha: %.2f, Energy: %.2f" % (alpha, energy))
			x.append(alpha)
			y.append(energy)

	if(rank == 0):

		t1 = time.time()
		print("Elapsed time: %.2f sec" % (t1 - t0))

		pyplot.xlabel("alpha")
		pyplot.ylabel("Energy")
		pyplot.plot(x, y, 'o', label="VMC")
		pyplot.legend()
		pyplot.show()











