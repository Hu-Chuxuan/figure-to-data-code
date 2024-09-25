import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import lsq_linear
import time

class ImageCleaner:
    def __init__(self):
        pass

    def cluster_by_neighbor(self, img):
        threshold = 3*2
        img = img.astype(np.int16)
        color_idx = np.zeros_like(img[:, :, 0], dtype=np.int32)-1
        cluster_centers = []
        idx2cnt = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if color_idx[i, j] != -1:
                    continue
                queue = [(i, j)]
                cur_colors = [img[i, j]]
                color_idx[i, j] = len(cluster_centers)
                colors = {tuple(img[i, j]): 1}
                while len(queue) > 0:
                    cur_i, cur_j = queue.pop(0)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            ni = cur_i + dx
                            nj = cur_j + dy
                            if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1] and color_idx[ni, nj] == -1 and \
                                np.sum(np.abs(img[ni, nj] - img[cur_i, cur_j])) < threshold:
                                queue.append((ni, nj))
                                color_idx[ni, nj] = color_idx[i, j]
                                if tuple(img[ni, nj]) in colors:
                                    colors[tuple(img[ni, nj])] += 1
                                else:
                                    colors[tuple(img[ni, nj])] = 1
                                    cur_colors.append(img[ni, nj])
                max_color = max(colors, key=colors.get)
                cluster_centers.append(np.array(max_color))
                idx2cnt.append(sum(colors.values()))
                
        print("Number of color clusters", len(cluster_centers))
        self.clustered = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.clustered[i, j] = cluster_centers[color_idx[i, j]]
        self.idx2color = color_idx
        self.cluster_centers = np.array(cluster_centers)
        self.idx2cnt = idx2cnt
    
    def build_neighbors(self):
        # find neighbors of each color
        self.color_idx2neighbors = {}
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        for i in range(self.clustered.shape[0]):
            for j in range(self.clustered.shape[1]):
                # Find the most frequent color among its neighbors
                neighbor_idx = []
                for k in range(4):
                    ni = i + dx[k]
                    nj = j + dy[k]
                    if 0 <= ni < self.clustered.shape[0] and 0 <= nj < self.clustered.shape[1] and not np.array_equal(self.clustered[ni, nj], self.clustered[i, j]):
                        neighbor_idx.append(self.color_idx[ni, nj])
                neighbor_idx = np.array(neighbor_idx)
                unique_idx = np.sort(np.unique(neighbor_idx))
                if len(unique_idx) < 2:
                    continue
                if self.color_idx[i, j] not in self.color_idx2neighbors:
                    self.color_idx2neighbors[self.color_idx[i, j]] = {}
                self.color_idx2neighbors[self.color_idx[i, j]][tuple(unique_idx)] = 1

    def build_candidates(self):
        self.idx2candidates = {}
        for idx in range(len(self.idx2color)):
            b = np.concatenate([self.idx2color[idx], np.ones((1))], axis=0)
            candidate_idx = set()
            if idx in self.color_idx2neighbors:
                for neighbors in self.color_idx2neighbors[idx]:
                    neighbor_colors = [self.idx2color[i].reshape(3, 1) for i in neighbors]
                    a = np.concatenate([np.concatenate(neighbor_colors, axis=1), np.ones((1, len(neighbor_colors)))], axis=0)
                    # x, residuals, rank, s = np.linalg.lstsq(a, b)
                    res = lsq_linear(a, b, bounds=(0, 1))
                    x = res.x
                    if res.success and np.allclose(a.dot(res.x), b) and 0 < np.max(x) < 1 and 0 < np.min(x) < 1 and np.abs(np.sum(x) - 1) < 1e-6:
                        for l in range(len(neighbors)):
                            if abs(x[l] - np.max(x)) < 1e-6:
                                candidate_idx.add(neighbors[l])
            self.idx2candidates[idx] = list(candidate_idx)
    
    def build_dependency_graph(self):
        self.color_mapping = {}
        self.dependencies = {}
        self.dependent_on = {}
        self.cnts = [0] * len(self.idx2color)
        # build the dependency graph
        for idx in range(self.idx2color.shape[0]):
            if idx in self.idx2candidates:
                self.idx2candidates[idx] = list(self.idx2candidates[idx])
                if len(self.idx2candidates[idx]) == 0:
                    self.color_mapping[idx] = idx
                    continue
                dependent = self.idx2candidates[idx][0]
                for candidate in self.idx2candidates[idx]:
                    if not np.array_equal(self.idx2color[candidate], self.idx2color[idx]) and \
                        np.sum(np.abs(self.idx2color[candidate] - self.idx2color[idx])) < np.sum(np.abs(self.idx2color[dependent] - self.idx2color[idx])):
                        dependent = candidate
                if dependent not in self.dependencies:
                    self.dependencies[dependent] = []
                self.dependencies[dependent].append(idx)
                self.dependent_on[idx] = dependent
                self.cnts[idx] = 1
            else:
                print("No dependency", idx)

    def topological_sort(self):
        queue = []
        for idx in range(len(self.cnts)):
            if self.cnts[idx] == 0:
                queue.append(idx)

        while len(queue) > 0:
            idx = queue.pop(0)
            if idx in self.dependencies:
                for dependent in self.dependencies[idx]:
                    self.cnts[dependent] -= 1
                    if self.cnts[dependent] == 0:
                        queue.append(dependent)
                        self.color_mapping[dependent] = self.color_mapping[self.dependent_on[dependent]]

        circle = set()
        for idx in range(len(self.cnts)):
            if self.cnts[idx] <= 0:
                continue
            queue = [idx]
            while len(queue) > 0:
                circle.add(queue[0])
                if queue[0] in self.dependencies:
                    for dependent in self.dependencies[queue[0]]:
                        self.cnts[dependent] -= 1
                        if self.cnts[dependent] == 0:
                            queue.append(dependent)
                self.cnts[self.dependent_on[queue[0]]] -= 1
                if self.cnts[self.dependent_on[queue[0]]] == 0:
                    queue.append(self.dependent_on[queue[0]])
                queue.pop(0)
            if len(circle) > 0:
                # print("Solving circle", circle)
                # set to the one has largest cnt
                majority_color = max(circle, key=lambda x: self.idx2cnt[x])
                for c in circle:
                    if c in self.color_mapping:
                        print("Conflict", c, "mapping to", self.color_mapping[c], "setting to", majority_color, "circle", circle)
                    self.color_mapping[c] = majority_color
                # print("Mapped circle to", majority_color)
                circle = set()

    def get_all_dependencies(self, idx, src_idx):
        if idx not in self.dependencies or len(self.dependencies[idx]) == 0:
            return set()
        result = set()
        for dependent in self.dependencies[idx]:
            if self.idx2cnt[dependent]*4 < self.idx2cnt[src_idx]:
                result.add(dependent)
                result = result.union(self.get_all_dependencies(dependent, src_idx))
        return result

    def get_dependent(self, idx, src_idx):
        # Search for the ones that idx depends on, change their colors if 
        # 1. dependent_on[idx] is less frequent than idx; 
        # 2. idx is the closest one among all more frequent colors dependent on dependent_on[idx]
        if idx in self.dependent_on and self.idx2cnt[self.dependent_on[idx]]*4 < self.idx2cnt[src_idx]:
            dependent = self.dependent_on[idx]
            result = set()
            deps = self.get_all_dependencies(dependent, src_idx)
            if len(deps) == 0:
                return result
            majority_color = max(deps, key=lambda x: self.idx2cnt[x])
            if majority_color == src_idx:
                result.add(dependent)
                result = result.union(self.get_dependent(dependent, src_idx))
            return result
        return set()

    def adjust_color_mapping(self):
        # Adjust color_mapping
        for idx in self.color_mapping:
            # 4 stands for up, down, left, right
            if self.idx2cnt[self.color_mapping[idx]]*4 < self.idx2cnt[idx]:
                # Change all colors depend on idx to the same color
                self.color_mapping[idx] = idx
                cluster_colors = self.get_all_dependencies(idx, idx)
                cluster_colors = cluster_colors.union(self.get_dependent(idx, idx))
                for c in cluster_colors:
                    self.color_mapping[c] = idx

    def process(self, img):
        self.cluster_by_neighbor(img)
        self.build_neighbors()
        self.build_candidates()
        self.build_dependency_graph()
        self.topological_sort()
        self.adjust_color_mapping()
        cluster_sharped = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                idx = self.color_idx[i, j]
                if idx in self.color_mapping:
                    cluster_sharped[i, j] = self.idx2color[self.color_mapping[idx]]
                else:
                    print("no mapping", idx)
        return cluster_sharped
    