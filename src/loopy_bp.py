from src.utils import *
import numpy as np
from numpy.linalg import norm
import cv2

class LoopyBP:
    # pixel arrangement:
    #              S
    #              |
    #        T ——— P ——— Q
    #              |
    #              R

    def __init__(self, left, right, max_disp=50, max_iter=60, tau=10, smooth_weight=10):
        self.max_disp = max_disp
        self.max_iter = max_iter
        self.tau = tau
        self.left = left.astype(float)
        self.right = right.astype(float)
        self.smooth_weight = smooth_weight

    def solve_bp(self):
        self.calc_data_cost()
        self.energy = np.zeros((self.max_iter))
        h, w = self.left.shape[:2]
        self.mS = np.zeros((h, w, self.max_disp))
        self.mR = np.zeros((h, w, self.max_disp))
        self.mT = np.zeros((h, w, self.max_disp))
        self.mQ = np.zeros((h, w, self.max_disp))
        
        for i in range(self.max_iter):
            print(f'Iteration {i}')
            self.update_messages()
            self.compute_beliefs()
            self.labels = np.argmin(self.beliefs, axis=2)
            self.energy[i] = self.calc_energy()
        
        return self.labels, self.energy


    def calc_data_cost(self):
        h, w = self.left.shape[:2]
        self.data_cost = np.zeros((h, w, self.max_disp))
        for d in range(self.max_disp):
            self.data_cost[:, : , d] = np.minimum(
                norm(self.left - np.roll(self.right, d, axis=1), axis=2, ord=1),
                    self.tau * np.ones((h, w))
                )
        return self.data_cost
    
    def calc_energy(self):
        h, w = self.left.shape[:2]
        hh, ww = np.meshgrid(range(h), range(w), indexing='ij')
        Ddp = self.data_cost[hh, ww, self.labels]
        E = np.sum(Ddp)
        int_cost_S = self.smooth_weight * (self.labels - np.roll(self.labels,  1, axis=0) != 0)
        int_cost_T = self.smooth_weight * (self.labels - np.roll(self.labels,  1, axis=1) != 0)
        int_cost_R = self.smooth_weight * (self.labels - np.roll(self.labels, -1, axis=0) != 0)
        int_cost_Q = self.smooth_weight * (self.labels - np.roll(self.labels, -1, axis=1) != 0)
        # set boundary costs to zero
        int_cost_S[0, :] = 0
        int_cost_T[:, 0] = 0
        int_cost_R[-1, :] = 0
        int_cost_Q[:, -1] = 0

        E += np.sum(int_cost_S) + np.sum(int_cost_T) \
                + np.sum(int_cost_R) + np.sum(int_cost_Q)
        return E
    
    def update_messages(self):
        h, w = self.data_cost.shape[:2]
        # initialize empty messages
        mS = np.zeros(self.data_cost.shape)
        mR = np.zeros(self.data_cost.shape)
        mT = np.zeros(self.data_cost.shape)
        mQ = np.zeros(self.data_cost.shape)
        # in_mX -- messages going in to X
        in_mS = np.roll(self.mR,  1, axis=0)
        in_mR = np.roll(self.mS, -1, axis=0)
        in_mT = np.roll(self.mQ,  1, axis=1)
        in_mQ = np.roll(self.mT, -1, axis=1)
        # npX -- neighbours excluding X
        npS = self.data_cost + in_mT + in_mR + in_mQ
        npR = self.data_cost + in_mS + in_mT + in_mQ
        npT = self.data_cost + in_mS + in_mR + in_mQ
        npQ = self.data_cost + in_mS + in_mR + in_mT

        spS = np.amin(npS, axis=2)
        spR = np.amin(npR, axis=2)
        spT = np.amin(npT, axis=2)
        spQ = np.amin(npQ, axis=2)

        for d in range(self.max_disp):
            self.mS[:, :, d] = np.minimum(npS[:, :, d], spS + self.smooth_weight)
            self.mR[:, :, d] = np.minimum(npR[:, :, d], spR + self.smooth_weight)
            self.mT[:, :, d] = np.minimum(npT[:, :, d], spT + self.smooth_weight)
            self.mQ[:, :, d] = np.minimum(npQ[:, :, d], spQ + self.smooth_weight)
        # normalization
        self.mS -= np.mean(self.mS, axis=2)[:, :, np.newaxis]
        self.mR -= np.mean(self.mR, axis=2)[:, :, np.newaxis]
        self.mT -= np.mean(self.mT, axis=2)[:, :, np.newaxis]
        self.mQ -= np.mean(self.mQ, axis=2)[:, :, np.newaxis]

    def compute_beliefs(self):
        self.beliefs = self.data_cost.copy()
        in_mS = np.roll(self.mR,  1, axis=0)
        in_mR = np.roll(self.mS, -1, axis=0)
        in_mT = np.roll(self.mQ,  1, axis=1)
        in_mQ = np.roll(self.mT, -1, axis=1)
        self.beliefs += in_mS + in_mR + in_mT + in_mQ
  




