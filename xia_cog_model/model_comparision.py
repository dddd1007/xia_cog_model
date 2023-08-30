# Import required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_additional_fit_metrics(model):
    additional_metrics = {}
    rsquared = model.rsquared
    adj_rsquared = model.rsquared_adj
    rmse = np.sqrt(model.mse_resid)
    mae = np.mean(np.abs(model.resid))
    durbin_watson = sm.stats.durbin_watson(model.resid)
    dic = 2 * model.df_model - 2 * model.llf
    aic_min = model.aic
    akaike_weight = np.exp(-0.5 * (model.aic - aic_min))

    return {
        "R-squared": rsquared,
        "Adjusted R-squared": adj_rsquared,
        "RMSE": rmse,
        "MAE": mae,
        "Durbin-Watson": durbin_watson,
        "DIC": dic,
        "Akaike weight": akaike_weight,
    }


def calculate_linear_model_fits(raw_data, pe_columns):
    fit_metrics_df = pd.DataFrame()
    for sub_num in raw_data["sub_num"].unique():
        sub_data = raw_data[raw_data["sub_num"] == sub_num]
        sub_data["run_num"] = sub_data["run_num"].astype(str)
        run_dummies = pd.get_dummies(sub_data["run_num"], prefix="run")
        fit_metrics_sub = {"sub_num": sub_num}

        for pe_col in pe_columns:
            X = pd.concat(
                [
                    sub_data[
                        [pe_col, "stim_loc_num", "resp_num", "volatility_num"]
                    ],
                    run_dummies,
                ],
                axis=1,
            )
            y = sub_data["rt"]
            model = sm.OLS(y, sm.add_constant(X, prepend=False)).fit()
            fit_metrics_sub.update(
                {
                    f"{pe_col}_loglikelihood": model.llf,
                    f"{pe_col}_AIC": model.aic,
                    f"{pe_col}_BIC": model.bic,
                    **{
                        f"{pe_col}_{key}": value
                        for key, value in calculate_additional_fit_metrics(
                            model
                        ).items()
                    },
                }
            )
        fit_metrics_df = fit_metrics_df.append(
            fit_metrics_sub, ignore_index=True
        )

    return fit_metrics_df


def reshape_fit_metrics_df(fit_metrics_df):
    desc_stats_df = fit_metrics_df.describe().transpose()
    mean_stats_df = desc_stats_df.loc[desc_stats_df.index.str.contains("_mean")]
    mean_stats_df["SpecificModel"] = (
        mean_stats_df.index.str.split("_").str[:-1].str.join("_")
    )
    mean_stats_df["FitMetric"] = mean_stats_df.index.str.split("_").str[-1]
    mean_stats_pivot_df = mean_stats_df.pivot(
        index="SpecificModel", columns="FitMetric", values="mean"
    )
    mean_stats_pivot_df.reset_index(inplace=True)
    return mean_stats_pivot_df


def calculate_exceedance_probability(bic_values):
    unnormalized_prob = np.exp(-0.5 * np.array(bic_values))
    exceedance_prob = unnormalized_prob / np.sum(unnormalized_prob)
    return exceedance_prob
