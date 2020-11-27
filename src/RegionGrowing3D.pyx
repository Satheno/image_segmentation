#include <Python.h>
from collections import deque
import numpy as np
cimport numpy as cnp
cimport cython

cdef class RegionGrowing3D:
    cdef int[:,:,:] img_3d
    cdef int[:,:,:] mask
    cdef int[:,:,:] out_mask
    cdef int axis_dim
    cdef int lower_threshold
    cdef int upper_threshold
    cdef int neighbor_count
    cdef queue

    def __cinit__(self, int[:,:,:] img_3d, int[:,:,:] mask, int neighbor_count, int lower_threshold, int upper_threshold):
        self.img_3d = img_3d
        self.axis_dim = len(img_3d)
        self.mask = mask
        self.neighbor_count = neighbor_count
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.out_mask = np.zeros_like(self.mask)
        self.queue = deque()

    def calculate_region(self, int[:] seed):
        cdef int[:] item = seed
        cdef cnp.ndarray[int, ndim=2] neighbors
        cdef int x
        cdef int y
        cdef int z

        self.out_mask[item[0], item[1], item[2]] = 1
        self.queue.append(np.array([item[0], item[1], item[2]]))

        while len(self.queue) != 0:
            item = self.queue.pop()
            if self.neighbor_count == 26:
                neighbors =  np.array([[item[0] - 1, item[1] - 1, item[2] - 1],
                             [item[0] - 1, item[1] - 1, item[2]],
                             [item[0] - 1, item[1] - 1, item[2] + 1],
                             [item[0] - 1, item[1], item[2] - 1],
                             [item[0] - 1, item[1], item[2]],
                             [item[0] - 1, item[1], item[2] + 1],
                             [item[0] - 1, item[1] + 1, item[2] - 1],
                             [item[0] - 1, item[1] + 1, item[2]],
                             [item[0] - 1, item[1] + 1, item[2] + 1],
                             [item[0], item[1] - 1, item[2] - 1],
                             [item[0], item[1] - 1, item[2]],
                             [item[0], item[1] - 1, item[2] + 1],
                             [item[0], item[1], item[2] - 1],
                             [item[0], item[1], item[2] + 1],
                             [item[0], item[1] + 1, item[2] - 1],
                             [item[0], item[1] + 1, item[2]],
                             [item[0], item[1] + 1, item[2] + 1],
                             [item[0] + 1, item[1] - 1, item[2] - 1],
                             [item[0] + 1, item[1] - 1, item[2]],
                             [item[0] + 1, item[1] - 1, item[2] + 1],
                             [item[0] + 1, item[1], item[2] - 1],
                             [item[0] + 1, item[1], item[2]],
                             [item[0] + 1, item[1], item[2] + 1],
                             [item[0] + 1, item[1] + 1, item[2] - 1],
                             [item[0] + 1, item[1] + 1, item[2]],
                             [item[0] + 1, item[1] + 1, item[2] + 1]])
            elif self.neighbor_count == 6:
                neighbors = np.array([[item[0], item[1], item[2] - 1],
                             [item[0], item[1], item[2] + 1],
                             [item[0], item[1] - 1, item[2]],
                             [item[0], item[1] + 1, item[2]],
                             [item[0] - 1, item[1], item[2]],
                             [item[0] + 1, item[1], item[2]]])
            for neighbor in neighbors:
                x = neighbor[0]
                y = neighbor[1]
                z = neighbor[2]
                if x < self.axis_dim and y < self.axis_dim and z < self.axis_dim and x > -1 and y > -1 and z > -1 and \
                        self.mask[x, y, z] == 1 and self.out_mask[x, y, z] == 0:
                    if self.is_acceptable(self.img_3d[x, y, z]):
                        self.out_mask[x, y, z] = 1
                        self.queue.append(np.array([x, y, z]))
        return np.array(self.out_mask)

    cdef is_acceptable(self, int value):
        return value < self.upper_threshold and value > self.lower_threshold
