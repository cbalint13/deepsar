#!/usr/bin/python

from __future__ import print_function

from caffe.proto import caffe_pb2
from caffe.coord_map import crop
from caffe import layers as L, params as P, to_proto

import sys
import math
import caffe


def bn_relu( mode, bottom, dropout=0 ):

    bottom = L.BatchReNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    bottom = L.Scale(bottom, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    bottom = L.ReLU(bottom, in_place=True)

    if (dropout>0):
      bottom = L.Dropout(bottom, in_place=True)

    return bottom

def unet(mode='train', batch_size=12, dropout=0.2):

    s = 2

    if (mode=='train'):
      data, label = L.Python(ntop=2, python_param=dict(module='h5datalayer', layer='H5DataLayer', param_str=str(dict(h5_file='hdf5/dataset-00000000.h5', batch_size=batch_size))) )

    if (mode=='tests'):
      data, label = L.HDF5Data(ntop=2, source="test.txt", batch_size=1)

    if (mode=='infer'):
      data = L.Input( ntop=1, input_param=dict(shape=dict(dim=[1,2,224,224])) )

    conv1 = L.Convolution(data,  convolution_param=dict(num_output=int(64/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv1 = bn_relu(mode,conv1)
    conv1 = L.Convolution(conv1, convolution_param=dict(num_output=int(64/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv1 = bn_relu(mode,conv1)
    pool1 = L.Pooling(conv1, kernel_size=2, pad=0, stride=2, pool=P.Pooling.MAX)

    conv2 = L.Convolution(pool1, convolution_param=dict(num_output=int(128/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv2 = bn_relu(mode,conv2)
    conv2 = L.Convolution(conv2, convolution_param=dict(num_output=int(128/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv2 = bn_relu(mode,conv2)
    pool2 = L.Pooling(conv2, kernel_size=2, pad=0, stride=2, pool=P.Pooling.MAX)

    conv3 = L.Convolution(pool2, convolution_param=dict(num_output=int(256/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv3 = bn_relu(mode,conv3)
    conv3 = L.Convolution(conv3, convolution_param=dict(num_output=int(256/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv3 = bn_relu(mode,conv3)
    pool3 = L.Pooling(conv3, kernel_size=2, pad=0, stride=2, pool=P.Pooling.MAX)

    conv4 = L.Convolution(pool3, convolution_param=dict(num_output=int(512/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv4 = bn_relu(mode,conv4)
    conv4 = L.Convolution(conv4, convolution_param=dict(num_output=int(512/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv4 = bn_relu(mode,conv4,dropout)
    #conv4 = L.Dropout(conv4, dropout_ratio=dropout)
    pool4 = L.Pooling(conv4, kernel_size=2, pad=0, stride=2, pool=P.Pooling.MAX)

    conv5 = L.Convolution(pool4, convolution_param=dict(num_output=int(1024/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv5 = bn_relu(mode,conv5)
    conv5 = L.Convolution(conv5, convolution_param=dict(num_output=int(1024/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    conv5 = bn_relu(mode,conv5,dropout)
    #conv5 = L.Dropout(conv5, dropout_ratio=dropout)

    up1 = L.Deconvolution(conv5, convolution_param=dict(num_output=int(512/s), kernel_size=2, pad=0, stride=2, group=int(512/s), bias_term=False, weight_filler=dict(type='bilinear'), bias_filler=dict(type='constant')) )
    up1 = bn_relu(mode,up1)

    crop1 = crop(conv4, up1)
    conc1 = L.Concat(up1, crop1)

    up2 = L.Convolution(conc1,  convolution_param=dict(num_output=int(512/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up2 = bn_relu(mode,up2)
    up2 = L.Convolution(up2,    convolution_param=dict(num_output=int(512/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up2 = bn_relu(mode,up2)

    up3 = L.Deconvolution(up2,  convolution_param=dict(num_output=int(256/s), kernel_size=2, pad=0, stride=2, group=int(256/s), bias_term=False, weight_filler=dict(type='bilinear'), bias_filler=dict(type='constant')) )
    up3 = bn_relu(mode,up3)

    crop2 = crop(conv3, up3)
    conc2 = L.Concat(up3, crop2)

    up4 = L.Convolution(conc2,  convolution_param=dict(num_output=int(256/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up4 = bn_relu(mode,up4)
    up4 = L.Convolution(up4,    convolution_param=dict(num_output=int(256/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up4 = bn_relu(mode,up4)

    up5 = L.Deconvolution(up4,  convolution_param=dict(num_output=int(128/s), kernel_size=2, pad=0, stride=2, group=int(128/s), bias_term=False, weight_filler=dict(type='bilinear'), bias_filler=dict(type='constant')) )
    up5 = bn_relu(mode,up5)

    crop3 = crop(conv2, up5)
    conc3 = L.Concat(up5, crop3)

    up6 = L.Convolution(conc3,  convolution_param=dict(num_output=int(128/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up6 = bn_relu(mode,up6)
    up6 = L.Convolution(up6,    convolution_param=dict(num_output=int(128/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up6 = bn_relu(mode,up6)

    up7 = L.Deconvolution(up6,  convolution_param=dict(num_output=int(128/s), kernel_size=2, pad=0, stride=2, group=int(128/s), bias_term=False, weight_filler=dict(type='bilinear'), bias_filler=dict(type='constant')) )
    up7 = bn_relu(mode,up7)

    crop4 = crop(conv1, up7)
    conc4 = L.Concat(up7, crop4)

    up8 = L.Convolution(conc4,  convolution_param=dict(num_output=int(64/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up8 = bn_relu(mode,up8)
    up8 = L.Convolution(up8,    convolution_param=dict(num_output=int(64/s), kernel_size=3, pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )
    up8 = bn_relu(mode,up8)

    model = L.Convolution(up8, convolution_param=dict(num_output=4, kernel_size=1, pad=0, stride=1, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')) )


    if (mode=='train'):

      loss = L.EuclideanLoss(model, label, loss_param=dict(normalization=2))
      #accu = L.Accuracy(model, label)

      return to_proto(loss)

    if (mode=='tests'):

      loss = L.EuclideanLoss(model, label, loss_param=dict(normalization=2))
      #accu = L.Accuracy(model, label)

      return to_proto(loss)

    if (mode=='infer'):

      return to_proto(model)


def make_net():

    with open('unet.prototxt', 'w') as f:
      print(str( unet(mode='train') ), file=f)

    with open('unet_tests.prototxt', 'w') as f:
      print(str( unet(mode='tests') ), file=f)

    with open('unet_infer.prototxt', 'w') as f:
      print(str( unet(mode='infer') ), file=f)

def make_solver():

    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE


    s.test_net.append('unet_tests.prototxt')
    s.test_interval = 50
    s.test_iter.append(500)
    s.test_initialization = True

    s.train_net = 'unet.prototxt'

    s.type = 'Adam'
    s.display = 1
    s.iter_size = 2
    s.max_iter = 500000
    s.snapshot = 500
    s.snapshot_format = s.HDF5
    s.snapshot_prefix = "./snapshots/unet"

    s.base_lr = 0.001
    s.momentum = 0.9
    s.weight_decay = 0.0001

    s.lr_policy='step'
    s.gamma = 0.75
    s.stepsize = 50000
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'unet_solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net()
    make_solver()
