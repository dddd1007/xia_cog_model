data {
  int<lower=1> N;    // trial number
  int corr_react[N]; // sub's correct reaction
  int space_loc[N];  // stim space location
}

parameters {
  real k;
  real<lower=0.01, upper=0.99> r_l[N]; // Probability for stim on left
  real<lower=0.01, upper=0.99> r_r[N]; // Probability for stim on right
  real v[N];
}

model {
  k~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v[t] ~ uniform(-100,100);
      r_l[t] ~ normal(0.5,0.45);
      r_r[t] ~ normal(0.5,0.45);
    }
    else{
      v[t] ~ normal(v[t-1],exp(k));
      if(space_loc[t] == 0){
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v[t]));
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v[t]));
        corr_react[t] ~ bernoulli(r_l[t]);
      }else if(space_loc[t] == 1){
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v[t]));
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v[t]));
        corr_react[t] ~ bernoulli(r_r[t]);
      }
    }
  }
}