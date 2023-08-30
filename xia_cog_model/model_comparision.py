import pandas as pd
import numpy as np
import statsmodels.api as sm


# Function to calculate additional fit metrics
def calculate_additional_fit_metrics(model):
    # Initialize an empty dictionary to store additional fit metrics
    additional_metrics = {}

    # R-squared
    rsquared = model.rsquared
    additional_metrics["R-squared"] = rsquared

    # Adjusted R-squared
    adj_rsquared = model.rsquared_adj
    additional_metrics["Adjusted R-squared"] = adj_rsquared

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(model.mse_resid)
    additional_metrics["RMSE"] = rmse

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(model.resid))
    additional_metrics["MAE"] = mae

    # Durbin-Watson statistic
    durbin_watson = sm.stats.durbin_watson(model.resid)
    additional_metrics["Durbin-Watson"] = durbin_watson

    # Deviance Information Criterion (DIC) - not directly available for OLS, so we approximate
    # Note: This is a rough approximation and should be interpreted cautiously
    dic = 2 * model.df_model - 2 * model.llf
    additional_metrics["DIC"] = dic

    # Akaike weights and Evidence ratio (from AIC)
    # Note: These metrics are more meaningful when comparing multiple models
    aic_min = model.aic
    akaike_weight = np.exp(-0.5 * (model.aic - aic_min))
    additional_metrics["Akaike weight"] = akaike_weight

    return additional_metrics


# Modify the function to include additional fit metrics
def calculate_linear_model_fits(raw_data, pe_columns):
    # Create an empty dataframe to store fit metrics
    fit_metrics_df = pd.DataFrame()
    fit_metrics_df = fit_metrics_df.dropna()

    # Loop over each subject
    for sub_num in raw_data["sub_num"].unique():
        # Filter data for the current subject
        sub_data = raw_data[raw_data["sub_num"] == sub_num]

        # Convert 'run_num' to string and create dummy variables for the current subject
        sub_data["run_num"] = sub_data["run_num"].astype(str)
        run_dummies = pd.get_dummies(sub_data["run_num"], prefix="run")

        # Create a dictionary to store fit metrics for the current subject
        fit_metrics_sub = {"sub_num": sub_num}

        # Loop over each PE column (each model's PE)
        for pe_col in pe_columns:
            # Prepare the independent variables
            X = pd.concat(
                [
                    sub_data[
                        [pe_col, "stim_loc_num", "resp_num", "volatility_num"]
                    ],
                    run_dummies,
                ],
                axis=1,
            )
            X = sm.add_constant(
                X, prepend=False
            )  # Add a constant but keep it at the end (run_num serves as the intercept)

            # Prepare the dependent variable
            y = sub_data["rt"]

            # Fit the linear model
            model = sm.OLS(y, X).fit()

            # Extract fit metrics
            loglikelihood = model.llf
            aic = model.aic
            bic = model.bic

            # Add basic fit metrics to the dictionary
            fit_metrics_sub[f"{pe_col}_loglikelihood"] = loglikelihood
            fit_metrics_sub[f"{pe_col}_AIC"] = aic
            fit_metrics_sub[f"{pe_col}_BIC"] = bic

            # Calculate additional fit metrics
            additional_metrics = calculate_additional_fit_metrics(model)
            for key, value in additional_metrics.items():
                fit_metrics_sub[f"{pe_col}_{key}"] = value

        # Add the dictionary to the dataframe
        fit_metrics_df = fit_metrics_df.append(
            fit_metrics_sub, ignore_index=True
        )

    return fit_metrics_df


def reshape_fit_metrics_df(fit_metrics_df):
    desc_stats_df = fit_metrics_df.describe().transpose()
    mean_stats_df = desc_stats_df.loc[desc_stats_df.index.str.contains("_mean")]
    mean_stats_df["SpecificModel"] = mean_stats_df["Metric"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    mean_stats_df["FitMetric"] = mean_stats_df["Metric"].apply(
        lambda x: x.split("_")[-1]
    )

    # Pivot the DataFrame
    mean_stats_pivot_df = mean_stats_df.pivot(
        index="SpecificModel", columns="FitMetric", values="Mean"
    )
    mean_stats_pivot_df.reset_index(inplace=True)
    return mean_stats_pivot_df


def calculate_exceedance_probability(bic_values):
    """
    Calculate the exceedance probability based on BIC values.
    Exceedance probability is calculated as exp(-0.5 * BIC) normalized by the sum for all models.

    Parameters:
    - bic_values (array-like): The BIC values for all models

    Returns:
    - exceedance_prob (array-like): The exceedance probabilities for all models
    """
    # Calculate the unnormalized exceedance probability
    unnormalized_prob = np.exp(-0.5 * np.array(bic_values))

    # Normalize the exceedance probabilities so they sum to 1
    exceedance_prob = unnormalized_prob / np.sum(unnormalized_prob)

    return exceedance_prob


# Extract BIC values from the fit_metrics DataFrame for each model
bic_values_by_model = fit_metrics_df_v2.filter(like="BIC").mean()

# Calculate exceedance probabilities
exceedance_probabilities = calculate_exceedance_probability(bic_values_by_model)

# Create a DataFrame to display the results
exceedance_prob_df = pd.DataFrame(
    {
        "Model": bic_values_by_model.index.str.replace("_BIC", ""),
        "Exceedance Probability": exceedance_probabilities,
    }
)

exceedance_prob_df.sort_values(by="Exceedance Probability", ascending=False)
