# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:00:45 2023

@author: Martin
"""
from tensorflow.keras import layers, regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K


class Custom_dense(layers.Layer):
   """
   Create a dense layer with tied weights
   This enforces the weights on decoder to be the same as on encoder
   """
   def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               freeze_weights=None,
               **kwargs):
      super().__init__(**kwargs)
      self.freeze_weights = freeze_weights
      self.units = units
      self.activation = activations.get(activation)
      self.use_bias = use_bias
      self.kernel_initializer = initializers.get(kernel_initializer)
      self.bias_initializer = initializers.get(bias_initializer)
      self.kernel_regularizer = regularizers.get(kernel_regularizer)
      self.bias_regularizer = regularizers.get(bias_regularizer)
      self.activity_regularizer = regularizers.get(activity_regularizer)
      self.kernel_constraint = constraints.get(kernel_constraint)
      self.bias_constraint = constraints.get(bias_constraint)
     
   def build(self, input_dimension):
      if self.freeze_weights is not None:
         self.kernel = K.transpose(self.freeze_weights.kernel)
         self._non_trainable_weights.append(self.kernel)
      else:
         self.kernel = self.add_weight(shape=(input_dim, self.units),
                               initializer=self.kernel_initializer,
                               name='kernel',
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)
      if self.use_bias:
         self.bias = self.add_weight(shape=(self.units,),
                            initializer=self.bias_initializer,
                            name='bias',
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
      else:
         self.bias = None
      self.built = True
      

   def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
     
      
