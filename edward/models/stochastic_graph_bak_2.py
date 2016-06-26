# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classes and helper functions for Stochastic Computation Graphs.

## Stochastic Computation Graph Classes

@@StochasticTensor
@@DistributionTensor

## Stochastic Computation Value Types

@@SampleValue
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import threading

import six

from tensorflow.contrib import distributions
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
#from tensorflow.python.platform import tf_logging as logging

STOCHASTIC_TENSOR_COLLECTION = "_stochastic_tensor_collection_"


@six.add_metaclass(abc.ABCMeta)
class StochasticTensor(object):
  """Base Class for Tensor-like objects that emit stochastic values."""

  def __init__(self, **kwargs):
    self._inputs = kwargs

  @abc.abstractproperty
  def name(self):
    pass

  @abc.abstractproperty
  def dtype(self):
    pass

  @abc.abstractmethod
  def value(self, name=None):
    pass

  @staticmethod
  def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      raise ValueError("%s: Ref type is not supported." % v)
    return v.value()


# pylint: disable=protected-access
ops.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._tensor_conversion_function)
# pylint: enable=protected-access


class _StochasticValueType(object):

  def pushed_above(self, unused_value_type):
    pass

  def popped_above(self, unused_value_type):
    pass

  @abc.abstractproperty
  def stop_gradient(self):
    """Whether the value should be wrapped in stop_gradient.

    StochasticTensors must respect this property.
    """
    pass


class SampleValue(_StochasticValueType):
  """Draw n samples along a new outer dimension.

  This ValueType draws `n` samples from StochasticTensors run within its
  context, increasing the rank by one along a new outer dimension.

  Example:

  ```python
  mu = tf.zeros((2,3))
  sigma = tf.ones((2, 3))
  with sg.value_type(sg.SampleValue(n=4)):
    dt = sg.DistributionTensor(
      distributions.Normal, mu=mu, sigma=sigma)
  # draws 4 samples each with shape (2, 3) and concatenates
  assertEqual(dt.value().get_shape(), (4, 2, 3))
  ```
  """

  def __init__(self, n=1, stop_gradient=False):
    """Sample `n` times and concatenate along a new outer dimension.

    Args:
      n: A python integer or int32 tensor. The number of samples to take.
      stop_gradient: If `True`, StochasticTensors' values are wrapped in
        `stop_gradient`, to avoid backpropagation through.
    """
    self._n = n
    self._stop_gradient = stop_gradient

  @property
  def n(self):
    return self._n

  @property
  def stop_gradient(self):
    return self._stop_gradient


class DistributionTensor(StochasticTensor):
  """The DistributionTensor is a StochasticTensor backed by a distribution.
  """

  def __init__(self, dist_cls, name=None, **dist_args):
    self._dist_cls = dist_cls
    self._dist_args = dist_args
    self._value_type = SampleValue()

    with ops.op_scope(dist_args.values(), name, "DistributionTensor") as scope:
      self._name = scope
      self._dist = dist_cls(**dist_args)
      self._value = self._create_value()

    super(DistributionTensor, self).__init__()

  def _create_value(self):
    """Create the value Tensor based on the value type, store as self._value."""

    if isinstance(self._value_type, SampleValue):
      value_tensor = self._dist.sample(self._value_type.n)
    else:
      raise TypeError(
          "Unrecognized Distribution Value Type: %s", self._value_type)

    stop_gradient = self._value_type.stop_gradient

    if stop_gradient:
      # stop_gradient is being enforced by the value type
      return array_ops.stop_gradient(value_tensor)

    if (isinstance(self._dist, distributions.ContinuousDistribution)
        and self._dist.is_reparameterized):
      return value_tensor  # Using pathwise-derivative for this one.
    else:
      # Will have to perform some variant of score function
      # estimation.  Call stop_gradient on the sampler just in case we
      # may accidentally leak some gradient from it.
      return array_ops.stop_gradient(value_tensor)

  #@property
  #def distribution(self):
  #  return self._dist

  #def clone(self, name=None, **dist_args):
  #  return DistributionTensor(self._dist_cls, name=name, **dist_args)

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dist.dtype

  #def entropy(self, name="entropy"):
  #  return self._dist.entropy(name=name)

  #def mean(self, name="mean"):
  #  return self._dist.mean(name=name)

  def value(self, name="value"):
    return self._value
