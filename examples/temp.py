#sg = tf.contrib.bayesflow.stochastic_graph
import edward as ed
sg = ed

distributions = tf.contrib.distributions
mu = [0.0, 0.1, 0.2]
sigma = tf.constant([1.1, 1.2, 1.3])
sigma2 = tf.constant([0.1, 0.2, 0.3])
prior = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)

sess = tf.Session()
sess.run(tf.identity(prior))
sess.run(prior + tf.constant(5.0))
sess.run([prior.mean(), prior.value()])

import edward as ed
import tensorflow as tf

#x = ed.RandomVariable()
x = ed.Normal(loc=tf.constant(0.0), scale=tf.constant(1.0))
x + tf.constant(5.0)

sess = tf.Session()
sess.run(x)
sess.run(tf.identity(x))
sess.run(x + tf.constant(5.0))
