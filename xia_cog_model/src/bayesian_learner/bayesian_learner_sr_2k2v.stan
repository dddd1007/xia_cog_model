data {
  int<lower=1> N;    // trial number
  int react[N];      // sub's correct reaction
  int space_loc[N];  // stim space location
}

parameters {
  real k_l;
  real k_r;
  real<lower=0.01, upper=0.99> r_l[N]; // Probability for stim on left
  real<lower=0.01, upper=0.99> r_r[N]; // Probability for stim on right
  real v_l[N]; // Volatile for r_l
  real v_r[N]; // Volatile for r_r
}

model {
  k_l~uniform(-10,10);
  k_r~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v_l[t] ~ uniform(-100,100);
      v_r[t] ~ uniform(-100,100);
      r_l[t] ~ normal(0.5,0.45);
      r_r[t] ~ normal(0.5,0.45);
    }
    else{
      if(space_loc[t] == 0){
        v_l[t] ~ normal(v_l[t-1],exp(k_l));
        v_r[t] ~ normal(v_r[t-1],exp(k_r));
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v_l[t]));
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v_r[t]));
        react[t] ~ bernoulli(r_l[t]);
      }else if(space_loc[t] == 1){
        v_l[t] ~ normal(v_l[t-1],exp(k_l));
        v_r[t] ~ normal(v_r[t-1],exp(k_r));
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v_l[t]));
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v_r[t]));
        react[t] ~ bernoulli(r_r[t]);
      }
    }
  }
}