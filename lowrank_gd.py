import sys
import numpy as np
import pandas as pd
from utils.data_utils import make_submission


def read(fn):
	with open(fn, 'r') as f:
		data = f.read()

	data = data.strip().split('\n')[1:]

	data = [x.split(',') for x in data]
	data = [[*x[0].split('_'), x[1]] for x in data]

	data = np.array(data).astype(int)
	return data


if __name__ == '__main__':

	seed = 0 if len(sys.argv) <= 1 else int(sys.argv[1])
	np.random.seed(seed)

	all_ratings = read('data/train_ratings.csv')

	Ns, Np = 10000, 1000

	A = np.zeros((Ns, Np))
	A[*all_ratings.T[:2]] = all_ratings[:, 2]

	# train:val split of 0.9:0.1
	split = np.random.binomial(1, 0.9, size=all_ratings.shape[0]).astype(bool)

	train = np.zeros((Ns, Np))
	train[*all_ratings.T[:2, split]] = 1

	val = np.zeros((Ns, Np))
	val[*all_ratings.T[:2, ~split]] = 1


	lr = 1
	D = 6
	num_iter = 10000

	# initialize embeddings
	U = np.random.randn(Ns, D) * 1e-3
	V = np.random.randn(Np, D) * 1e-3

	best = (np.inf, U, V, -1)

	for it in trange(num_iter):
		# compute prediction error
		err = U @ V.T - A
		train_err = err * train

		# find gradient (normalized)
		ds = train_err @ V / Np
		dp = train_err.T @ U / Ns
		
		# update embeddings
		U -= lr * ds
		V -= lr * dp
		
		# stop GD if validation rmse did not decrease in 500 iterations
		val_err = err * val
		val_rmse = np.sqrt(np.sum(val_err ** 2) / val.sum())
		if best[0] > val_rmse:
			best = (val_rmse, U, V, it)
			patience = 500
		elif patience > 0:
			patience -= 1
		else:
			break

	rmse, U, V, it = best
	print(D, 'dims : ValRMSE =', rmse.round(4), '@ iter', it + 1)

	pred = U @ V.T
	make_submission(lambda i,j: pred[i, j], f'lowrank{D}_gd_seed{seed}.csv')

#################

# Logs

# [seed=0] 6 dims : ValRMSE = 0.8673 @ iter 410
# [seed=1] 6 dims : ValRMSE = 0.8683 @ iter 399
# [seed=2] 6 dims : ValRMSE = 0.8684 @ iter 390
# [seed=3] 6 dims : ValRMSE = 0.8631 @ iter 404
# [seed=4] 6 dims : ValRMSE = 0.8721 @ iter 375
# [seed=5] 6 dims : ValRMSE = 0.8624 @ iter 5836

#################
