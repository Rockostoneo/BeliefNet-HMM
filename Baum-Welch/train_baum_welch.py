from hmmlearn import hmm
import h5py as h5
import time
import sys
sys.stderr.flush()

def save_model(model, train_loss, eval_loss, idx, params):
    filename = f"bw_d{params['d']}_m{params['m']}_lamda{params['lamda']}_iters{params['n_iter']}_fit{idx+1}.h5"
    with h5.File(filename, 'x') as file:
        file.create_dataset('A', data=model.transmat_)
        file.create_dataset('C', data=model.emissionprob_)
        file.create_dataset('pi', data=model.startprob_)
        file.create_dataset('train_loss', data=train_loss)
        file.create_dataset('eval_loss', data=eval_loss)
        file.create_dataset('params', data=str(params))

def get_pretrained_parameters(filename):
    with h5.File(filename, 'r') as f:
        A = f['A'][:]
        C = f['C'][:]
        pi = f['pi'][:]
    d = A.shape[0]
    return A, C, pi, d

def train_hmm_models(training_set, validation_set, params, seed_offset, n_iter, n_fits, d, pretrained_matricies=None):
    if pretrained_matricies is not None:
        A, C, pi = pretrained_matricies

    start_time = time.time()
    print(f'Starting training with {n_fits} fits...')
    for idx in range(n_fits):
        fit_start = time.time()
        if pretrained_matricies is not None:
            model = hmm.CategoricalHMM(
                n_components=d, random_state=idx+seed_offset,
                init_params='',verbose = True, n_iter=n_iter)
            model.startprob_ = pi
            model.transmat_ = A
            model.emissionprob_ = C
        else: 
            model = hmm.CategoricalHMM(
                n_components=d, random_state=idx+seed_offset,
                init_params='ste',verbose = True, n_iter=n_iter)

        X = training_set.flatten()
        Y = validation_set.flatten()
        x_lengths = [training_set.shape[1]] * training_set.shape[0]
        y_lengths = [validation_set.shape[1]] * validation_set.shape[0]
        
        model.fit(X.reshape(-1,1), x_lengths)
        fit_end = time.time()
        
        train_loss = -model.score(X.reshape(-1,1), x_lengths) / X.size
        eval_loss = -model.score(Y.reshape(-1,1), y_lengths) / Y.size

        print(
            f'Model #{idx}\ttrain_loss: {train_loss:.4f}\teval_loss: {eval_loss:.6f}'
            f'\tTime: {fit_end - fit_start:.2f}s'
        )

        # Store all models and scores
        save_model(model, train_loss, eval_loss, idx, params)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'Average time per model: {total_time/n_fits:.2f} seconds')   

if __name__ == '__main__':
    d = 64  # number of hidden states
    n_fits = 5 # number of fits with different initializations
    n_iter = 20  # number of iterations for each fit
    seed_offset = 123
    #A, C, pi, d = get_pretrained_parameters('hmm_learned_matricies.h5') #if starting from pretrained matricies, otherwise will be randomly initialized by hmmlearn
    #pretrained_matricies = (A, C, pi)
    pretrained_matricies = None #train from random initialization

    with h5.File('../prepare_data/data_d64_m32_lamda0.9_example.h5', 'r') as f:
        params = eval(f['params'][()])  #retrieve parameters used to generate matricies
        training_set = f['z_train'][:]
        validation_set = f['z_validation'][:]
    
    params['seed_offset'] = seed_offset
    params['n_fits'] = n_fits
    params['n_iter'] = n_iter
    
    train_hmm_models(training_set, validation_set, params, seed_offset, n_iter, n_fits, d, pretrained_matricies=pretrained_matricies)