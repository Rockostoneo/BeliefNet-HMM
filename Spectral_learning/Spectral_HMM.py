import numpy as np
import h5py as h5

class Spectral_HMM:
    def __init__(self, n_observations, n_hidden):
        self.P1 = np.zeros((n_observations),dtype=float)  
        self.P21 = np.zeros((n_observations, n_observations),dtype=float) 
        self.P3x1 = np.zeros((n_observations, n_observations, n_observations),dtype=float)  
        self.n_hidden = n_hidden
        self.n_observations = n_observations

    def train(self, observations):  #observations is n x L
        trilst = np.array([sequence[idx: idx+3] for sequence in observations
                           for idx in range(len(sequence)-2)], dtype=int)
        for obs_seq in trilst:
            self.P1[obs_seq[0]] += 1 
            self.P21[obs_seq[1], obs_seq[0]] += 1 
            self.P3x1[obs_seq[1], obs_seq[2], obs_seq[0]] += 1  #remember x is first
        
        self.P1 = self.P1 / np.sum(self.P1)
        self.P21 = self.P21 / np.sum(self.P21)
        self.P3x1 = self.P3x1 / np.sum(self.P3x1) 

        U, S, V = np.linalg.svd(self.P21)
        U = U[:, 0:self.n_hidden]  # m x d
        factor = np.linalg.pinv(np.dot(self.P21.T, U))
        self._b1 = np.dot(U.T, self.P1)        
        self._binf = np.dot(factor, self.P1)
        self._Bx = np.zeros((self.n_observations, self.n_hidden, self.n_hidden), dtype=float)        
        for index in range(self.n_observations):
            tmp = np.dot(U.T, self.P3x1[index])
            self._Bx[index] = np.dot(tmp,factor.T)

    def predict_next(self, observation): #observation is L, returns a non normalized probability vector of next observations (L-1,m)  
        bt = np.zeros((len(observation),self.n_hidden), dtype=float)
        bt[0] = self._b1
        for t in range(1, len(observation)):
            temp = (self._binf.T @ self._Bx[observation[t-1]] @ bt[t-1])
            if temp == 0:
                bt[t] = self._b1
                print("Warning: temp is zero, resetting bt[t] to b1")
            else:    
                bt[t] = (self._Bx[observation[t-1]] @ bt[t-1]) / temp
        predicted_obs_probs = np.zeros((len(observation)-1, self.n_observations), dtype=float) #start after the initial observation
        for t in range(len(observation)-1):
            tmp = np.matmul(self._Bx, bt[t+1]) 
            temp = np.vecdot(self._binf.T,tmp)  
            predicted_obs_probs[t] = temp / np.sum(temp)
            
            #if any predicted probabilities are negative, set them to the minimum positive probability in the vector
            min_positive = np.min(predicted_obs_probs[t][predicted_obs_probs[t]>0])  
            predicted_obs_probs[t][predicted_obs_probs[t] <= 0] = min_positive
            predicted_obs_probs[t] = predicted_obs_probs[t] / np.sum(predicted_obs_probs[t])  #renormalize


        return predicted_obs_probs

    def save_model(self, file_path):
        with h5.File(file_path, 'x') as f:
            f.create_dataset('P1', data=self.P1)
            f.create_dataset('P21', data=self.P21)
            f.create_dataset('P3x1', data=self.P3x1)
            f.create_dataset('n_hidden', data=self.n_hidden)
            f.create_dataset('n_observations', data=self.n_observations)