# 导入所需的库
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logsumexp


# 定义函数 calculate_linear_model_fits，用于计算线性模型的拟合度
def calculate_linear_model_fits(
    raw_data,  # 原始数据
    pe_columns=["bl_sr_pe", "bl_ab_pe", "rl_sr_v_pe", "rl_ab_v_pe"],  # PE 列
    sub_col="sub_num",  # 子列
    dep_var=["rt"],  # 依赖变量
    indep_var=["stim_loc_num", "resp_num", "volatility_num"],  # 独立变量
    interception_col="run_num",  # 截距列
    debug=False,  # 是否打印调试信息
):
    """
    Calculate the fit of the linear model.

    Parameters:
        - raw_data: Original data
        - pe_columns: PE columns
        - sub_col: Sub-column
        - dep_var: Dependent variable
        - indep_var: Independent variable
        - interception_col: Interception column

    Returns:
        - fit_metrics_df: DataFrame containing fit metrics
    """
    # 创建一个副本以避免 SettingWithCopyWarning
    filtered_data = raw_data.dropna().copy()
    df_list = []
    # 将所有需要的列都转换为浮点数类型
    for col in pe_columns + dep_var:
        filtered_data[col] = filtered_data[col].astype("float64")
    for col in indep_var:
        filtered_data[col] = filtered_data[col].astype("category")
    # 对每个子列进行操作
    for sub_num in sorted(filtered_data[sub_col].unique()):
        sub_data = filtered_data[filtered_data[sub_col] == sub_num].copy()
        sub_data.loc[:, interception_col] = sub_data[interception_col].astype(str)
        # 创建虚拟变量
        run_dummies = pd.get_dummies(
            sub_data[interception_col], prefix="inter_"
        ).astype(int)

        # 对每个 PE 列进行操作
        for pe_col in pe_columns:
            X = pd.concat(
                [
                    sub_data[[pe_col, *indep_var]],
                    run_dummies,
                ],
                axis=1,
            )
            y = sub_data[dep_var].copy()
            # 拟合模型
            if debug:
                print(f"=== 正在处理的子列：{sub_num}, PE 列：{pe_col} ===")
                print("X 的前五行数据：")
                print(X.head())
                print("y 的数据：")
                print(y)
            model = sm.OLS(y, X).fit()
            # 计算额外的拟合度指标
            fit_metrics_sub = calculate_additional_fit_metrics(model)
            fit_metrics_sub["sub_num"] = sub_num
            fit_metrics_sub["model_type"] = pe_col
            fit_metrics_sub_df = pd.DataFrame([fit_metrics_sub])
            df_list.append(fit_metrics_sub_df)

    # 合并所有的拟合度指标
    fit_metrics_df = pd.concat(df_list, ignore_index=True)

    return fit_metrics_df


# 定义函数 calculate_additional_fit_metrics，用于计算额外的拟合度指标
def calculate_additional_fit_metrics(model):
    """
    Calculate additional fit metrics.

    Parameters:
        - model: Fitted model
        - pe_col: PE column

    Returns:
        - A dictionary containing all fit metrics
    """
    rsquared = model.rsquared
    adj_rsquared = model.rsquared_adj
    rmse = np.sqrt(model.mse_resid)
    mae = np.mean(np.abs(model.resid))
    durbin_watson = sm.stats.durbin_watson(model.resid)
    dic = 2 * model.df_model - 2 * model.llf
    aic_min = model.aic
    akaike_weight = np.exp(-0.5 * (model.aic - aic_min))

    # 返回一个字典，包含了所有的拟合度指标
    return {
        "loglikelihood": model.llf,
        "AIC": model.aic,
        "BIC": model.bic,
        "R-squared": rsquared,
        "Adjusted R-squared": adj_rsquared,
        "RMSE": rmse,
        "MAE": mae,
        "Durbin-Watson": durbin_watson,
        "DIC": dic,
        "Akaike weight": akaike_weight,
    }


# 定义函数 summary_model_comparison，用于对模型进行比较
def summary_model_comparison(fit_metrics_df):
    """
    Compare models.

    Parameters:
        - fit_metrics_df: DataFrame containing fit metrics

    Returns:
        - mean_stats_pivot_df: DataFrame containing average statistics for each model type
    """
    # 计算每种模型类型的平均统计量
    mean_stats_pivot_df = fit_metrics_df.groupby("model_type").mean()
    return mean_stats_pivot_df


# 定义函数 calculate_exceedance_probability_df，用于计算超越概率
def calculate_exceedance_probability_df(data, metrics):
    """
    Calculate exceedance probability using random effects Bayesian model comparison and return as a DataFrame.

    Parameters:
        - data: DataFrame containing the metric values of each model and topic
        - metrics: List of metrics used for model comparison (e.g., ['loglikelihood', 'AIC', 'BIC'])

    Returns:
        - exceedance_prob_df: DataFrame containing the exceedance probability of each metric and model
    """
    # 初始化一个空的 DataFrame 来存储超越概率
    exceedance_prob_df = pd.DataFrame()

    for metric in metrics:
        # 初始化变量
        num_models = len(data["model_type"].unique())
        num_subjects = len(data["sub_num"].unique())

        # 检查观察值的数量是否大于模型中的参数数量
        if num_subjects <= num_models:
            print("警告：观察值的数量小于或等于模型中的参数数量。")
            continue

        # 计算每个模型的平均值和方差
        mean_values = data.groupby("model_type")[metric].mean()
        var_values = data.groupby("model_type")[metric].var()

        # 检查是否存在完全共线的情况
        if var_values.min() == 0:
            print("警告：存在完全共线的情况。")
            continue

        # 初始化每个模型的 alpha（Dirichlet 参数）为 1
        alpha = np.ones(num_models)

        # 计算每个模型和主题的预期对数模型证据
        log_evidence = -0.5 * (
            np.log(2 * np.pi * var_values) + (mean_values**2) / var_values
        )

        # 计算每个模型的组对数模型证据
        group_log_evidence = logsumexp(log_evidence)

        # 计算超越概率
        exceedance_prob = np.exp(log_evidence - group_log_evidence)

        # 将此度量的超越概率添加到 DataFrame
        exceedance_prob_df[metric] = exceedance_prob

    # 转置 DataFrame，使度量为行，模型为列
    exceedance_prob_df = exceedance_prob_df.T
    exceedance_prob_df.reset_index(inplace=True)
    exceedance_prob_df.rename(columns={"index": "Metric"}, inplace=True)

    return exceedance_prob_df
