// Bayesian Observer model from Behrens et al. (2007)
// Implements hazard-rate based change-point detection in Bernoulli reward sequence

// The agent assumes that on each trial the underlying reward probability may reset
// to a uniform prior with probability h (hazard rate). Otherwise it remains the same.
// Belief about the reward probability is represented as a Beta distribution with
// parameters alpha and beta. Prior to each observation, the predicted probability is
// the mean of this Beta distribution. After observing the outcome, the parameters are
// updated according to the hazard rate.

// data
data {
  int<lower=1> T;                       // number of trials
  int<lower=0,upper=1> y[T];            // binary outcomes
}

// parameters
parameters {
  real<lower=0,upper=1> h;              // hazard rate of environmental change
}

// transformed parameters
transformed parameters {
  real<lower=0> alpha[T + 1];           // Beta shape parameter for successes
  real<lower=0> beta[T + 1];            // Beta shape parameter for failures
  real<lower=0,upper=1> p[T];           // predicted reward probability before trial t

  // initialize with uniform prior Beta(1,1)
  alpha[1] = 1;
  beta[1] = 1;

  for (t in 1:T) {
    // predictive probability before observing y[t]
    p[t] = alpha[t] / (alpha[t] + beta[t]);

    // posterior update after observing outcome y[t]
    alpha[t + 1] = (1 - h) * (alpha[t] + y[t]) + h * 1;
    beta[t + 1]  = (1 - h) * (beta[t] + 1 - y[t]) + h * 1;
  }
}

// model
model {
  // uniform prior on hazard rate
  h ~ beta(1, 1);

  // likelihood of observations given predictive probabilities
  for (t in 1:T) {
    y[t] ~ bernoulli(p[t]);
  }
}
