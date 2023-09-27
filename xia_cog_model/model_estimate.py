import cmdstanpy
from julia import Main

bl_model_dir = "./src/bayesian_learner"
bl_model_type = "sr_1k1v_neg_v"

bl_model_loc = os.path.join(bl_model_dir, bl_model_type)
estimate_bl_model
