data {
  int<lower=1> N; // trial number
  array[N] int y;
}

parameters {
  real k;
  array[N] real<lower=0.01, upper=0.99> r;
  array[N] real v;
}

model {
  k~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v[t] ~ uniform(0,100);
      r[t] ~ normal(0.5,0.45);
    }
    else{
      v[t] ~ normal(v[t-1],exp(k));
      r[t] ~ beta_proportion(r[t-1],exp(v[t]));
    }
    y[t] ~ bernoulli(r[t]);
  }
}
