#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Olexa Bilaniuk, Chiheb Trabelsi, Sandeep Subramanian

# Imports
import complexnn
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D,\
                                                    ComplexConv3D,\
                                                    ComplexDense,\
                                                    FFT,IFFT,FFT2,IFFT2,\
                                                    SpectralPooling1D,SpectralPooling2D,ComplexFRN
from complexnn import GetImag, GetReal,Spline,CReLU,CRot,DenseHess
 
import h5py                                  as     H
import tensorflow.keras
from   tensorflow.keras.callbacks                       import Callback, ModelCheckpoint, LearningRateScheduler
from   tensorflow.keras.datasets                        import cifar10, cifar100
from   tensorflow.keras.initializers                    import Orthogonal
from   tensorflow.keras.layers                          import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D,Multiply
from   tensorflow.keras                          import Model
from   tensorflow.keras.optimizers                      import SGD, Adam, RMSprop
from   tensorflow.keras.preprocessing.image             import ImageDataGenerator
from   tensorflow.keras.regularizers                    import l2
from   tensorflow.keras.utils                  import to_categorical
import tensorflow.keras.backend                         as     K
import tensorflow.keras.models                          as     KM
try:
    from   kerosene.datasets                     import svhn2
except:
    print("SVHN dataset not available")
import logging                               as     L
import numpy                                 as     np
import os, pdb, socket, sys, time
import tensorflow as tf
#import theano                                as     T


#
# Residual Net Utilities
#

def learnConcatRealImagBlock(I, filter_size, featmaps, stage, block, convArgs, bnArgs, d):
	"""Learn initial imaginary component for input."""
	
	conv_name_base = 'res'+str(stage)+block+'_branch'
	bn_name_base   = 'bn' +str(stage)+block+'_branch'

	#O = BatchNormalization(name=bn_name_base+'2a', **bnArgs)(I)
	#O = Activation(d.act)(O)
	O = Convolution2D(featmaps[0], filter_size,
	                  name               = conv_name_base+'2a',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(I)
	
	O = BatchNormalization(name=bn_name_base+'2b', **bnArgs)(O)
	O = Activation(d.act)(O)
	O = Convolution2D(featmaps[1], filter_size,
	                  name               = conv_name_base+'2b',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(O)
	
	return O

def getResidualBlock(I, filter_size, featmaps, stage, block, shortcut, convArgs, bnArgs, d):
	"""Get residual block."""
	
	activation           = d.act
	drop_prob            = d.dropout
	nb_fmaps1, nb_fmaps2 = featmaps
	conv_name_base       = 'res'+str(stage)+block+'_branch'
	bn_name_base         = 'bn' +str(stage)+block+'_branch'
	#if K.image_data_format() == 'channels_first' and K.ndim(I) != 3:
	#	channel_axis = 1
	#else:
	channel_axis = -1
	
	
	if   d.model == "real":
		O = BatchNormalization(name=bn_name_base+'_2a', **bnArgs)(I)
	elif d.model == "complex":
		O = ComplexBN(name=bn_name_base+'_2a', **bnArgs)(I)
		O = CReLU()(O)  # Activation(activation)(O)

	#Activation(activation)(O)

	if shortcut == 'regular' or d.spectral_pool_scheme == "nodownsample":
		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
	elif shortcut == 'projection':
		if d.spectral_pool_scheme == "proj":
			O = applySpectralPooling(O, d)
		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
	if   d.model == "real":
		O = BatchNormalization(name=bn_name_base+'_2b', **bnArgs)(O)
		O = CReLU()(O)#Activation(activation)(O)
		O = Conv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
	elif d.model == "complex":
		O = ComplexBN(name=bn_name_base + '_2b', **bnArgs)(O)
		O = CReLU()(O)  # Activation(activation)(O)
		O = ComplexConv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)

	if   shortcut == 'regular':
		O = Add()([O, I])
	elif shortcut == 'projection':
		if d.spectral_pool_scheme == "proj":
			I = applySpectralPooling(I, d)
		if   d.model == "real":
			X = Conv2D(nb_fmaps2, (1, 1),
			           name    = conv_name_base+'1',
			           strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                     (1, 1),
			           **convArgs)(I)
			O      = Concatenate(channel_axis)([X, O])
		elif d.model == "complex":
			X = ComplexConv2D(nb_fmaps2, (1, 1),
			                  name    = conv_name_base+'1',
			                  strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                            (1, 1),
			                  **convArgs)(I)

			O_real = Concatenate(channel_axis)([X[...,:X.shape[-1]//2],O[...,:O.shape[-1]//2]])#Add()([X[...,:X.shape[-1]//2],O[...,:O.shape[-1]//2]])
			O_imag = Concatenate(channel_axis)([X[...,X.shape[-1]//2:],O[...,O.shape[-1]//2:]])#Add()([X[...,X.shape[-1]//2:],O[...,O.shape[-1]//2:]])
			O      = Concatenate(channel_axis)([O_real,     O_imag])

	return O

def applySpectralPooling(x, d):
	"""Perform spectral pooling on input."""

	if d.spectral_pool_gamma > 0 and d.spectral_pool_scheme != "none":
		x = FFT2 ()(x)
		x = SpectralPooling2D(gamma=(d.spectral_pool_gamma,
		                             d.spectral_pool_gamma))(x)
		x = IFFT2()(x)
	return x

def complex_max_pooling(x):
    x_real = x[...,:x.shape[-1]//2]
    x_imag = x[...,x.shape[-1]//2:]
    abs_val = tf.sqrt(x_real**2+x_imag**2)
    x_mask = tf.where(abs_val==tf.reduce_max(abs_val,axis=[1,2],keepdims=True),tf.ones_like(abs_val),tf.zeros_like(abs_val))
    shp = x.shape
    tile_shp = [1]*len(shp)+[2]
    x_mask = tf.tile(tf.expand_dims(x_mask,axis=-1),tile_shp)
    x_last = tf.stack([x_real,x_imag],axis=-1)
    x_max = x_last*x_mask
    x_max = tf.reduce_sum(x_max,axis=[1,2],keepdims=True)
    return tf.concat([x_max[...,0],x_max[...,1]],axis=-1)


#
# Get ResNet Model
#

def getResnetModel(d):
	n             = d.num_blocks
	sf            = d.start_filter
	dataset       = d.dataset
	activation    = d.act
	advanced_act  = d.aact
	drop_prob     = d.dropout
	inputShape    = (32, 32, 3)#(3, 32, 32) if K.image_dim_ordering() == "th" else (32, 32, 3)
	channelAxis   = -1#1 if K.image_data_format() == 'channels_first' else -1
	filsize       = (3, 3)
	convArgs      = {
		"padding":                  "same",
		"use_bias":                 False,
		"kernel_regularizer":       l2(0.0001),
	}
	bnArgs        = {
		"axis":                     channelAxis,
		"momentum":                 0.9,
		"epsilon":                  1e-04
	}

	if   d.model == "real":
		sf *= 2
		convArgs.update({"kernel_initializer": Orthogonal(float(np.sqrt(2)))})
	elif d.model == "complex":
		convArgs.update({"spectral_parametrization": d.spectral_param,
						 "kernel_initializer": d.comp_init})


	#
	# Input Layer
	#

	I = tf.keras.Input(shape=inputShape)

	#
	# Stage 1
	#

	O = learnConcatRealImagBlock(I, (1, 1), (3, 3), 0, '0', convArgs, bnArgs, d)

	O = Concatenate(channelAxis)([I, O])
	if d.model == "real":
		O = Conv2D(sf, filsize, name='conv1', **convArgs)(O)
		O = BatchNormalization(name="bn_conv1_2a", **bnArgs)(O)
	else:
		O = ComplexBN(name="bn_conv1_2a", **bnArgs)(O)
		#O = CReLU()(O)
		O = ComplexConv2D(sf, filsize, name='conv1', **convArgs)(O)

#Activation(activation)(O)
	
	#
	# Stage 2
	#
	
	for i in range(n):
		O = getResidualBlock(O, filsize, [sf, sf], 2, str(i), 'regular', convArgs, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)
	
	#
	# Stage 3
	#
	O = getResidualBlock(O, filsize, [sf, sf], 3, '0', 'projection', convArgs, bnArgs, d)
	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)
	
	for i in range(n-1):
		O = getResidualBlock(O, filsize, [sf*2, sf*2], 3, str(i+1), 'regular', convArgs, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)
	
	#
	# Stage 4
	#
	
	O = getResidualBlock(O, filsize, [sf*2, sf*2], 4, '0', 'projection', convArgs, bnArgs, d)
	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)
	
	for i in range(n-1):
		O = getResidualBlock(O, filsize, [sf*4, sf*4], 4, str(i+1), 'regular', convArgs, bnArgs, d)
		if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
			O = applySpectralPooling(O, d)
	
	#
	# Pooling
	#
	
	if d.spectral_pool_scheme == "nodownsample":
		O = applySpectralPooling(O, d)
		O = AveragePooling2D(pool_size=(32, 32))(O)
	else:

		O = AveragePooling2D(pool_size=(8,  8))(O)#complex_max_pooling(O)
	#
	# Flatten
	#
	#O = ComplexBN(name="output_bn", **bnArgs)(O)
	O = Flatten()(O)
	
	#
	# Dense
	#
	
	if   dataset == 'cifar10':
		#O = ComplexDense(10,  activation=None, kernel_regularizer=l2(0.0001))(O)
		#O = tf.abs(tf.complex(O[...,:O.shape[-1]//2],O[...,O.shape[-1]//2:]))#O = tf.where(O[...,:O.shape[-1]//2]>0,tf.math.atan(O[...,O.shape[-1]//2:]/O[...,:O.shape[-1]//2]),tf.zeros_like(O[...,:O.shape[-1]//2]))
		O = Dense(10, activation=None, kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'cifar100':
		O = Dense(100, activation=None, kernel_regularizer=l2(0.0001))(O)
	elif dataset == 'svhn':
		O = Dense(10,  activation=None, kernel_regularizer=l2(0.0001))(O)
	else:
		raise ValueError("Unknown dataset "+d.dataset)
	
	# Return the model
	return Model(inputs=I,outputs=O,name="resnet")


