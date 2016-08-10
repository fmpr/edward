library('rstan')

# tell stan to use all the CPU cores
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# fix random seed for reproducibility of results
set.seed(1)

# sample artificial sin-like time-series
N = 50
source = 2*sin(seq(0, 6, length=N)) + 1
noise = rnorm(N, 0, 0.2) 
obs = source + noise

# pachage the data for stan
data = list(
  N = N,
  y = obs
)

# compile the model (takes a minute or so)
model = rstan::stan_model(file='kf_simple_invgamma.stan')

# run inference
vb = TRUE # if false use MCMC
if (vb) {
  out = rstan::vb(
    object = model,
    data = data,
    algorithm = 'meanfield',
    init = 'random',
    iter = 5e3, # default = 10000
    tol_rel_obj = 0.001, # default = 0.01
    grad_samples = 10, # default = 1
    elbo_samples = 100, # default = 100
    output_samples = 10000, # default = 1000
    adapt_engaged = FALSE, # default = TRUE
    eta = 0.2, # default = ADAPTIVE
    seed = 1
  )
} else {
  sampling_iterations = 5e3
  out = rstan::sampling(
    object = model,
    data = data,
    chains = 4,
    algorithm = 'NUTS',
    iter = sampling_iterations,
    warmup = sampling_iterations/2,
    refresh = sampling_iterations/10, # show an update every n iterations
    seed = 1
  )
}

# extract results
print(out)
x_hat = apply(rstan::extract(out, pars="x")$x, 2, mean)
x_var = apply(rstan::extract(out, pars="x")$x, 2, var)

# plot estimated latent variables (smoothed time-series)
ts.plot(obs,col="red")
lines(source,col="black")
lines(x_hat,col="blue")
lines(x_hat + sqrt(x_var),col="blue",lty=2)
lines(x_hat - sqrt(x_var),col="blue",lty=2)


