import h5py as h5
from generate_data_utils import generate_base_matricies, data_from_matricies

#HMM parameters
lamda = 0.9 # A = lamda * A_circle + (1-lamda) * A_noise, controls how much noise is added to the cyclic transition matrix
d = 64 #hidden state dimension
m = 128 #observation dimension
matrix_seed = 34534
Atemp = 0.1
Ctemp = 0.01
min_sparsity_factor = 16 #controls sparcity of noise matricies

#Data generation parameters
n_train = 4000
n_validation = 400
n_test = 400
L = 256 #length of each sequence
data_seed = 4234

data_filename = f"data_d{d}_m{m}_lamda{lamda}.h5"

#A is a cyclic transition matrix with noise, C is a random emission matrix, pi is a uniform initial distribution
A, C, pi = generate_base_matricies(d=d, m=m, lamda=lamda, seed=matrix_seed, Atemp=Atemp, Ctemp=Ctemp, min_sparsity_factor=min_sparsity_factor)
x, z = data_from_matricies(A, C, pi, n_train=n_train, n_validation=n_validation, n_test=n_test, L=L, data_seed=data_seed)

data_dict = {
    'd': d,
    'm': m,
    'lamda': lamda,
    'matrix_seed': matrix_seed,
    'Atemp': Atemp,
    'Ctemp': Ctemp,
    'min_sparsity_factor': min_sparsity_factor,
    'data_seed': data_seed,
    'n_train': n_train,
    'n_validation': n_validation,
    'n_test': n_test,
    'L': L
}

with h5.File(data_filename, 'x') as hf:
    hf.create_dataset('x_full', data=x)
    hf.create_dataset('x_train', data=x[:n_train])
    hf.create_dataset('x_validation', data=x[n_train:n_train+n_validation])
    hf.create_dataset('x_test', data=x[n_train+n_validation:])
    hf.create_dataset('z_full', data=z)
    hf.create_dataset('z_train', data=z[:n_train])
    hf.create_dataset('z_validation', data=z[n_train:n_train+n_validation])
    hf.create_dataset('z_test', data=z[n_train+n_validation:])
    hf.create_dataset('A', data=A)
    hf.create_dataset('C', data=C)
    hf.create_dataset('pi', data=pi)
    hf.create_dataset('params', data=str(data_dict))

print(f"Generated data file: {data_filename}")