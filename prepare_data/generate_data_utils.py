import numpy as np
import random
from tqdm import tqdm

def get_probability_matrix(nrows: int, ncols: int, temperature: float = 1.0, min_sparsity_factor: int = 16):
    min_number_of_non_zero_elements = max(2, ncols // min_sparsity_factor)  
    probability_matrix = np.empty((nrows, ncols))
    for r in range(nrows):
        probability_vector = np.zeros(ncols)

        # Decide how many non-zero elements to generate
        number_of_non_zero_elements = np.random.randint(
            min_number_of_non_zero_elements, ncols + 1
        )

        # Generate the indices of the non-zero elements
        indices = np.random.choice(
            ncols, number_of_non_zero_elements, replace=False
        )

        # Generate the values of the non-zero elements
        values = np.random.rand(number_of_non_zero_elements)
        values = softmax(values, temperature)

        # Assign the values to the corresponding indices
        probability_vector[indices] = values

        # Assign the probability vector to the probability matrix
        probability_matrix[r, :] = probability_vector
    return probability_matrix

def softmax(vector, temperature: float = 1.0):
    _vector = vector/temperature
    exp_vector = np.exp(_vector-np.max(_vector))
    normalized_vector = exp_vector / np.sum(exp_vector)
    return normalized_vector

def generate_base_matricies(d: int, m:int, lamda:float, seed:int, Atemp:float = 0.1, Ctemp:float = 0.01, min_sparsity_factor: int = 16):
    np.random.seed(seed)
    A_circle = np.roll(np.diag(np.ones(d)), 1, axis=1)  # create a cyclic transition matrix
    A_noise = get_probability_matrix(d,d, temperature=Atemp, min_sparsity_factor=min_sparsity_factor)  
    A = lamda * A_circle + (1-lamda) * A_noise
    C = get_probability_matrix(d, m, temperature=Ctemp, min_sparsity_factor=min_sparsity_factor) 
    pi = np.ones(d)/d  #uniform initial distribution
    return A, C, pi

def data_from_matricies(A, C, pi, n_train:int, n_validation:int, n_test:int, L:int, data_seed:int):
    n = n_train + n_validation + n_test
    random.seed(data_seed)

    x = np.zeros((n, L), dtype=int)
    z = np.zeros((n, L), dtype=int)

    for i in tqdm(range(n), desc="Generating sequences"):
        for j in range(L):
            if j == 0:
                x[i,j] = random.choices(range(A.shape[0]), weights=pi)[0]
                z[i,j] = random.choices(range(C.shape[1]), weights=C[x[i,0]])[0]
            else:
                x[i,j] = random.choices(range(A.shape[0]), weights=A[x[i,j-1]])[0]
                z[i,j] = random.choices(range(C.shape[1]), weights=C[x[i,j]])[0]

    return x, z 
