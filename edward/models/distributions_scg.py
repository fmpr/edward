#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/bayesflow/python/kernel_tests/stochastic_graph_test.py
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/bayesflow/python/ops/stochastic_graph.py#L120
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distributions/python/ops/normal.py
#https://github.com/tensorflow/tensorflow/blob/9e8a4938c06301cc3afa4cd3a3c6c2279e81c98f/tensorflow/python/framework/constant_op.py
import tensorflow as tf

from tensorflow.python.framework import ops

class RandomVariable(object):
    """
    Random variable object whose instantiation is fundamentally random
    variable-based endowed with distributional methods.

    Parameters
    ----------
    shape : int or list, optional
        Shape of random variable(s). For multivariate distributions,
        the outermost dimension denotes the multivariate dimension.
    dtype : dtype, optional
         Type of elements stored in this tensor.
    """
    def __init__(self, shape=1, dtype=tf.float32):
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return "<ed.RandomVariable '{}' shape={} dtype={}>".format(
            self.name(), self.get_shape(), self.dtype.name)

    def __str__(self):
        return "<ed.RandomVariable '{}' shape={} dtype={}>".format(
            self.name(), self.get_shape(), self.dtype.name)

    def get_shape(self):
        if isinstance(self.shape, int):
            return ()

        return self.shape

    def name(self):
        return "hello"

    #def cdf(self):
    #def entropy(self):
    #def log_cdf(self):
    #def mean(self):
    #def mode(self):
    #def pdf(self): # or pmf
    #def std(self):
    #def variance(self):

    def log_pdf(self): # or log_pmf
        pass

    def sample_noise(self, size=1):
        raise NotImplementedError()

    def reparam(self, eps):
        raise NotImplementedError()

    def sample(self, size=1):
        return self.reparam(self.sample_noise(size))

    def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
        """Tensor is one realization of the random variable(s), and
        with the same shape."""
        _ = name
        if dtype and not dtype.is_compatible_with(v.dtype):
            raise ValueError(
              "Incompatible type conversion requested to type '%s' for variable "
              "of type '%s'" % (dtype.name, v.dtype.name))

        if as_ref:
            raise ValueError("%s: Ref type is not supported." % v)

        return tf.squeeze(v.sample())

ops.register_tensor_conversion_function(
    RandomVariable, RandomVariable._tensor_conversion_function)
# TODO
# + how to recognize and let samples from random variable object when
# its a parameter argument
# + doesn't support direct arithmetic on floats but supports
# + all operations form another distribution object. how to do that
# instead of forming another tensor? how to tell about calculating
# induced densities when forming another distribution object?

class Normal(RandomVariable):
    """
    p(x | params ) = prod_{i=1}^d Normal(x[i] | loc[i], scale[i])
    where params = {loc, scale}.
    """
    def __init__(self, num_factors=1, loc=None, scale=None, dtype=tf.float32):
        super(Normal, self).__init__(num_factors, dtype)
        self.num_factors = self.shape
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = True

        if loc is None:
            loc = tf.Variable(tf.random_normal([self.num_vars]))

        if scale is None:
            scale_unconst = tf.Variable(tf.random_normal([self.num_vars]))
            scale = tf.nn.softplus(scale_unconst)

        self.loc = loc
        self.scale = scale

    #def __str__(self):
    #    sess = get_session()
    #    m, s = sess.run([self.loc, self.scale])
    #    return "mean: \n" + m.__str__() + "\n" + \
    #           "std dev: \n" + s.__str__()

    def sample_noise(self, size=1):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        return tf.random_normal((size, self.num_vars))

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        return self.loc + eps * self.scale

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        loci = self.loc[i]
        scalei = self.scale[i]
        return norm.logpdf(xs[:, i], loci, scalei)

    def entropy(self):
        return tf.reduce_sum(norm.entropy(scale=self.scale))
