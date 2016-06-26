from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import distributions
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

class DistributionTensor(object):
    def __init__(self, dist_cls, name=None, **dist_args):
        self._dist_cls = dist_cls
        self._name = name
        self._dist_args = dist_args

        self._dist = dist_cls(**dist_args)
        self.sample_tensor = False
        value_tensor = self._dist.sample(1)
        if self.sample_tensor:
            self._value = value_tensor
        else:
            self._value = array_ops.stop_gradient(value_tensor)

        super(DistributionTensor, self).__init__()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dist.dtype

    def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            raise ValueError(
              "Incompatible type conversion requested to type '%s' for variable "
              "of type '%s'" % (dtype.name, v.dtype.name))

        if as_ref:
            raise ValueError("%s: Ref type is not supported." % v)

        return v._value

ops.register_tensor_conversion_function(
    DistributionTensor, DistributionTensor._tensor_conversion_function)
