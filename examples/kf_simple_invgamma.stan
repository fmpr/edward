data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  vector[N] x;
  real mu;
  real beta;
  real<lower=0> noise_proc;
  real<lower=0> noise_obs;
}
model {
  mu ~ normal(1,10);
  beta ~ normal(1,10);
  
  //noise_proc ~ cauchy(0,2.5);
  noise_proc ~ inv_gamma(1,1);
  //noise_obs ~ cauchy(0,2.5);
  noise_obs ~ inv_gamma(1,1);

  x[1] ~ normal(mu, 10); // fat prior on x
  for (n in 2:N)
    x[n] ~ normal(mu + beta * x[n-1], noise_proc);

  y ~ normal(x, noise_obs);
}
