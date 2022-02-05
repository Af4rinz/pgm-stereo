import sys
import time
import numpy as np
import cv2
import maxflow
from utils import pair2grey

class Graph_Cut_Solver:
    OCCLUDED_LABEL = 1
    MAX_ITER = 4

    def __init__(self, left, right, 
    search_depth=50, max_steps=-1, occ_cost=-1, 
    smooth_cost_lo=-1, smooth_cost_hi=-1, smooth_thresh=10):
        self.left, self.right = pair2grey(left, right)
        self.img_shape = self.left.shape
        self.img_indices = np.indices(self.img_shape)
        self.img_size = self.left.size

        self.search_depth = search_depth
        self.occ_cost = occ_cost
        self.smooth_cost_lo = smooth_cost_lo if smooth_cost_lo > 0 else occ_cost * 0.2
        self.smooth_cost_hi = smooth_cost_hi if smooth_cost_hi > 0 else occ_cost * 3
        self.smooth_thresh = smooth_thresh
        self.energy = float('inf')

        self.max_steps = max_steps if max_steps > 0 else search_depth
        self.search_interval = (search_depth // self.max_steps) + bool(search_depth % self.max_steps)
        self.search_steps = -1 * np.arange(0, search_depth + 1, self.search_interval)[::-1]
        rank = np.empty(len(self.search_steps), dtype=np.int)
        rank[np.argsort(self.search_steps)] = np.arange(len(self.search_steps))
        self.label_rank = dict(zip(self.search_steps, rank))

        # construct neighbours and edges
        indices = np.indices(self.img_shape)
        n1p = indices[:, 1: , :].reshape(2, -1)
        n1q = n1p + [[-1] , [0]]
        n2p = indices[:, :, :-1].reshape(2, -1)
        n2q = n2p + [[0] , [1]]
        self.neighbours = np.array([np.concatenate([n1p, n2p], axis=1) , np.concatenate([n1q, n2q], axis=1)])
        self.neighbours_roll = list(np.rollaxis(self.neighbours, 1))

        idx_p , idx_q = self.neighbours
        left_diff = self.left[list(idx_p)] - self.left[list(idx_q)]
        self.is_left_under = np.abs(left_diff) < self.smooth_thresh

    def solve(self):
        self.labels = np.full(self.img_shape, self.OCCLUDED_LABEL, dtype=np.int)
        label_done = np.zeros(len(self.search_steps), dtype=np.bool)

        for i in range(self.MAX_ITER):
            start = time.time()
            label_order = np.random.permutation(self.search_steps)
            print(label_order)
            for label in label_order:
                label_idx = self.label_rank[label]
                if label_done[label_idx]:
                    continue
                is_expanded = self.expand_label(label)
                if is_expanded:
                    label_done[:] = False
                label_done[label_idx] = True
                # all labels are fully expanded
            print('Iteration {} time: {}'.format(i, time.time() - start))
            if label_done.all():
                break
        return -1 * self.labels
    
    def expand_label(self, label):
        is_expanded = False
        G = maxflow.Graph[int](2 * self.img_size, 12 * self.img_size)

        self.add_E_data_occ(G, label)
        self.add_E_smooth(G, label)
        self.add_E_unique(G, label)
        E = G.maxflow() + self.E_data_occ

        if E < self.energy:
            self.update_labels(G, label)
            is_expanded = True
        self.energy = E
        return is_expanded
    

    def add_E_data_occ(self, G, label):
        y_idx, x_idx = self.img_indices
        is_label = self.labels == label
        is_occluded = self.labels == self.OCCLUDED_LABEL
        idx_shifted = np.where(is_occluded, x_idx, x_idx + self.labels)
        # ACTIVE NODES
        active_ssd = np.square(self.left - self.right[y_idx, idx_shifted]) - self.occ_cost
        active_ssd[is_label | is_occluded] =  -self.occ_cost - 1
        active_nodes = np.zeros(self.img_shape, dtype=np.int)
        active_nodes[is_label] = -1
        active_nodes[is_occluded] = -2
        is_node_active = np.logical_not(is_label | is_occluded)

        E_data_occ = active_ssd[is_label].sum()
        # LABEL NODES
        is_occluded = np.logical_not(self._is_in_img(x_idx + label))
        idx_shifted = np.where(is_occluded, x_idx, x_idx + label)
        label_ssd = np.square(self.left - self.right[y_idx, idx_shifted]) - self.occ_cost
        label_ssd[is_label | is_occluded] =  -self.occ_cost - 1
        label_nodes = np.zeros(self.img_shape, dtype=np.int)
        label_nodes[is_label] = -1
        label_nodes[is_occluded] = -2
        is_node_label = np.logical_not(is_label | is_occluded)

        node_count = is_node_label.sum() + is_node_active.sum()
        node_ids = G.add_nodes(node_count)
        node_idx = 0

        for r, c in np.ndindex(self.img_shape):
            if is_node_active[r, c]:
                node_id = node_ids[node_idx]
                active_nodes[r, c] = node_id
                node_idx += 1
                active_cost = active_ssd[r, c]
                G.add_tedge(node_id, 0, active_cost)
            if is_node_label[r, c]:
                node_id = node_ids[node_idx]
                label_nodes[r, c] = node_id
                node_idx += 1
                label_cost = label_ssd[r, c]
                G.add_tedge(node_id, label_cost, 0)
        self.is_node_active = is_node_active
        self.is_node_label = is_node_label
        self.active_nodes = active_nodes
        self.label_nodes = label_nodes
        self.E_data_occ = E_data_occ
    
    def add_E_smooth(self, G, label):
        p_labels, q_labels = self.labels[self.neighbours_roll]
        penalty_label = self.get_smooth_penalty(label)
        penalty_active_p = self.get_smooth_penalty(p_labels)
        penalty_active_q = self.get_smooth_penalty(q_labels)
        p_idx, q_idx = self.neighbours
        is_p_in_img = self._is_in_img(p_idx[1, :] + q_labels)
        is_q_in_img = self._is_in_img(q_idx[1, :] + p_labels)

        for nidx in range(self.neighbours.shape[2]):
            y_idx, x_idx = self.neighbours.T[nidx]
            p_label, q_label = self.labels[y_idx, x_idx]
            node_label_p, node_label_q = self.label_nodes[y_idx, x_idx]
            node_active_p, node_active_q = self.active_nodes[y_idx, x_idx]
            is_p_active, is_q_active = self.is_node_active[y_idx, x_idx]

            if node_label_p != -2 and node_label_q != -2:
                penalty = penalty_label[nidx]
                if node_label_p != -1 and node_label_q != -1:
                    self.add_smooth_weights(G, node_label_p, node_label_q, 0, penalty, penalty, 0)
                elif node_label_p != -1:
                    G.add_tedge(node_label_p, 0, penalty)
                elif node_label_q != -1:
                    G.add_tedge(node_label_q, 0, penalty)
            penalty_p, penalty_q = penalty_active_p[nidx], penalty_active_q[nidx]

            if p_label == q_label:
                if not is_p_active or not is_q_active:
                    continue
                self.add_smooth_weights(G, node_active_p, node_active_q, 0, penalty_p, penalty_q, 0)
                continue
            if is_p_active and is_q_in_img[nidx]:
                G.add_tedge(node_active_p, 0, penalty_p)
            if is_q_active and is_p_in_img[nidx]:
                G.add_tedge(node_active_q, 0, penalty_q)
    
    def add_E_unique(self, G, label):
        h, w = self.img_shape
        y_idx, x_idx = self.img_indices
        idx_shifted = x_idx + self.labels - label
        is_valid_shift = self._is_in_img(idx_shifted)
        idx_shifted = np.clip(idx_shifted, 0, w - 1)
        banned = self.is_node_active & is_valid_shift
        ban_label = self.label_nodes[y_idx, idx_shifted][banned]
        ban_active = self.active_nodes[banned]

        self.add_unique_weights(G, ban_label, ban_active)
        is_node_label = self.label_nodes != -2
        banned = self.is_node_active & is_node_label
        self.add_unique_weights(G, self.label_nodes[banned], self.active_nodes[banned])

    def add_unique_weights(self, G, sinks, sources):
        for i in range(sources.size):
            G.add_edge(sources[i], sinks[i], int('854775807'), 0)
    
    def get_smooth_penalty(self, labels):
        p_idx, q_idx = self.neighbours
        if type(labels) is np.ndarray:
            labels = labels[self.is_left_under]
        smoothness = np.full(p_idx.shape[1], self.smooth_cost_lo, dtype=np.float)
        p_idx_shifted, is_p_in_img = self._shift(p_idx[:, self.is_left_under], labels)
        q_idx_shifted, is_q_in_img = self._shift(q_idx[:, self.is_left_under], labels)
        diff_right = self.right[list(p_idx_shifted)] - self.right[list(q_idx_shifted)]
        is_left_under = self.is_left_under.copy()
        is_left_under[is_left_under] = np.logical_not(is_p_in_img & is_q_in_img)
        smoothness[is_left_under] = 0
        return smoothness
    
    def add_smooth_weights(self, G, n1, n2, w1, w2, w3, w4):
        G.add_tedge(n1, w4, w2)
        G.add_tedge(n2, 0, (w1 - w2))
        G.add_edge(n1, n2, 0, w3 - w4 - (w1 - w2))
    
    def update_labels(self, G, label):
        is_node_active = np.copy(self.is_node_active)
        if is_node_active.any():
            active_nodes = self.active_nodes[is_node_active]
            is_node_active[is_node_active] = G.get_grid_segments(active_nodes)
            self.labels[is_node_active] = self.OCCLUDED_LABEL
        is_node_label = np.copy(self.is_node_label)
        if is_node_label.any():
            label_nodes = self.label_nodes[is_node_label]
            is_node_label[is_node_label] = G.get_grid_segments(label_nodes)
            self.labels[is_node_label] = label

    def _is_in_img(self, idx):
        is_in_img = (0 <= idx) & (idx < self.img_shape[1])
        return is_in_img
    
    def _shift(self, idx, shift):
        h, w = self.img_shape
        idx_shifted = np.copy(idx)
        idx_shifted[1, :] += shift
        is_in_img = self._is_in_img(idx_shifted[1, :])
        idx_shifted[1, :] = np.clip(idx_shifted[1, :], 0, w - 1)
        return idx_shifted, is_in_img









