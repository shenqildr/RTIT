#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import h5py
import numpy as np
import torch
from numpy import (array, unravel_index, nditer, linalg, random, subtract)
from sklearn import preprocessing
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import random
from scipy.spatial import cKDTree


def load_data_cls(partition):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_cls_scanobject(partition):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'scanobject', 'object_only1')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, '%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        for key in f.keys():
            print(f[key].name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def scale_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2/3, high=1.5, size=[3])
    #xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    scale_pointcloud = np.multiply(pointcloud, xyz1).astype('float32')
    return scale_pointcloud

def rotate_perturbation_point_cloud(data):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    #angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    angles = np.random.uniform(low=0, high=360, size=[3])
    angles = angles*np.pi/180                              
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(data, R).astype(np.float32)

    return rotated_data



def transform_point_cloud_to_comparative_orthogonal_coordinate(pointcloud):
    """ transform point cloud to comparative distance between three fixed position(gravity center,the closest and farthest points to gravity center
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, comparative orthogonal coordinates
    """
    N=pointcloud.shape[0]
    corrdinate_data = pointcloud.copy()
    gravity_centre = np.mean(corrdinate_data, axis=0)      
    comparative_gravity = corrdinate_data - gravity_centre
    comparative_gravity_distance = linalg.norm(comparative_gravity, axis=-1).reshape(N, 1)

    further_point_index = comparative_gravity_distance.argmax()
    further_point=corrdinate_data[further_point_index]

    close_point_index = comparative_gravity_distance.argmin()
    close_point=corrdinate_data[close_point_index]

    vector_normal=np.cross(close_point,further_point)
    close_point_update=np.cross(further_point,vector_normal)

    #normalization
    further_point=further_point/linalg.norm(further_point)
    close_point_update=close_point_update/linalg.norm(close_point_update)
    vector_normal=vector_normal/linalg.norm(vector_normal)

    further_point=further_point.reshape(3,1)
    vector_normal=vector_normal.reshape(3,1)    
    close_point_update=close_point_update.reshape(3,1)

    Transform_matrix = np.concatenate((further_point, vector_normal, close_point_update), axis=1) 
    pc_np = np.dot(comparative_gravity, Transform_matrix)

    return pc_np


def transform_PCA(pointcloud):

    """ PCA transform method
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, normalized coordinates
    """
    pca = PCA(n_components=3)
    pca.fit(pointcloud)
    X_new = pca.transform(pointcloud)

    return X_new


def random_matrix_transformation(pointcloud):
    '''
    generate g=[[a,0,0],[0,b,0],[0,0,c]] a=b=c=+-1
    Input:
         Nx3 point cloud
    Return:
         Nx3 Transformed point cloud  
    '''
    a = 1 if random.random() < 0.5 else -1
    b = 1 if random.random() < 0.5 else -1
    c = 1 if random.random() < 0.5 else -1
    g = np.zeros((3,3))
    g[0][0] = a
    g[1][1] = b
    g[2][2] = c

    pointcloud = np.dot(pointcloud, g)

    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition
        self.opt = opt        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            if self.opt.translation:
                pointcloud = translate_pointcloud(pointcloud)
            if self.opt.jitter:
                pointcloud = jitter_pointcloud(pointcloud)

        if self.partition == 'test':
            if self.opt.test_rot_perturbation:
                pointcloud = rotate_perturbation_point_cloud(pointcloud)
                
        if self.opt.rtit == 'cat':
            pointcloud = transform_point_cloud_to_comparative_orthogonal_coordinate(pointcloud)
        else:
            pointcloud = pointcloud
    
        pointcloud = torch.from_numpy(pointcloud.astype(np.float32))

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanobjectNN(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        self.data, self.label = load_data_cls_scanobject(partition)
        self.num_points = num_points
        self.partition = partition
        self.opt = opt        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            if self.opt.translation:
                pointcloud = translate_pointcloud(pointcloud)
            if self.opt.jitter:
                pointcloud = jitter_pointcloud(pointcloud)

        if self.partition == 'test':
            if self.opt.test_rot_perturbation:
                pointcloud = rotate_perturbation_point_cloud(pointcloud)

        if self.opt.rtit == 'cat':
            pointcloud = transform_point_cloud_to_comparative_orthogonal_coordinate(pointcloud)
        else:
            pointcloud = pointcloud

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    '''
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

    trainval = ShapeNetPart(2048, 'trainval')
    test = ShapeNetPart(2048, 'test')
    data, label, seg = trainval[0]
    print(data.shape)
    print(label.shape)
    print(seg.shape)

    train = S3DIS(4096)
    test = S3DIS(4096, 'test')
    data, seg = train[0]
    print(data.shape)
    print(seg.shape)
    '''
    a = 1 if random.random() < 0.5 else -1
    b = 1 if random.random() < 0.5 else -1
    c = 1 if random.random() < 0.5 else -1
    g = np.zeros((3,3))
    g[0][0] = a
    g[1][1] = b
    g[2][2] = c

    print('matrix g ', g)

