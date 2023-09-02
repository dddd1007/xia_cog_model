# 导入所需的库
from os import getcwd
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import logsumexp


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


# 定义函数 calculate_linear_model_fits，用于计算线性模型的拟合度
def calculate_linear_model_fits(
    raw_data: pd.DataFrame,  # 原始数据
    pe_columns: list[str] = [
        "bl_sr_pe",
        "bl_ab_pe",
        "rl_sr_v_pe",
        "rl_ab_v_pe",
    ],  # PE 列
    sub_col: list[str] = ["sub_num"],  # 子列
    dep_var: list[str] = ["rt"],  # 依赖变量
    indep_var: list[str] = ["stim_loc_num", "resp_num", "volatility_num"],  # 独立变量
    interception_col: list[str] = ["run_num"],  # 截距列
    debug: bool = False,  # 是否打印调试信息
):
    """
    计算线性模型的拟合度。

    参数：
        - raw_data: 原始数据，类型为 DataFrame
        - pe_columns: PE 列，类型为字符串列表
        - sub_col: 子列，类型为字符串列表
        - dep_var: 依赖变量，类型为字符串列表
        - indep_var: 独立变量，类型为字符串列表
        - interception_col: 截距列，类型为字符串列表
        - debug: 是否打印调试信息，类型为布尔值

    返回：
        - fit_metrics_df: 包含拟合度指标的 DataFrame
    """
    # 创建一个副本以避免 SettingWithCopyWarning
    raw_data = raw_data[pe_columns + sub_col + dep_var + indep_var + interception_col]
    filtered_data = raw_data.dropna().copy()
    df_list = []
    # 将所有需要的列都转换为浮点数类型
    for col in pe_columns:
        filtered_data[col] = filtered_data[col].astype("float64")
    for col in indep_var:
        filtered_data[col] = filtered_data[col].astype("int")
    # 对每个子列进行操作
    for sub_num in sorted(filtered_data[sub_col[0]].unique()):
        sub_data = filtered_data[filtered_data[sub_col[0]] == sub_num].copy()
        sub_data.loc[:, interception_col[0]] = sub_data[interception_col[0]].astype(str)
        # 创建虚拟变量
        run_dummies = pd.get_dummies(
            sub_data[interception_col[0]], prefix="inter_"
        ).astype("int")

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

            # 检查观察值的数量是否大于模型中的参数数量
            if X.shape[0] <= X.shape[1]:
                print(f"警告：观察值的数量小于或等于模型中的参数数量。被试编号：{sub_num}")
                X.to_csv(f"/error_report/sub_{sub_num}.csv")
                print("The error file have saved to" + getcwd())

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
            fit_metrics_sub[f"beta_{pe_col}"] = model.params[pe_col]
            fit_metrics_sub[f"pvalue_{pe_col}"] = model.pvalues[pe_col]
            fit_metrics_sub_df = pd.DataFrame([fit_metrics_sub])
            df_list.append(fit_metrics_sub_df)

    # 合并所有的拟合度指标
    fit_metrics_df = pd.concat(df_list, ignore_index=True)

    return fit_metrics_df


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


def summary_model_beta(model_selection_results, save_path, figure_size=(10, 5)):
    """
    This function extracts each beta of each model and performs descriptive statistics
    from model_selection_results.

    It also performs a one-sample t-test for each column and saves the
    results to a csv file.

    Finally, it plots a boxplot of the results.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import numpy as np

    beta_df = pd.DataFrame()
    beta_columns = [
        col for col in model_selection_results.columns if col.startswith("beta_")
    ]
    for col_name in beta_columns:
        beta_df[col_name] = model_selection_results[col_name].dropna().abs().to_list()
    beta_df["sub_num"] = model_selection_results["sub_num"].unique()

    # Perform descriptive statistics
    desc_stats = beta_df.describe()
    desc_stats.to_csv(os.path.join(save_path, "beta_desc_stats.csv"))

    # Perform one-sample t-test for each column and save the results to a csv file
    t_test_results = {}
    for col in beta_df.columns:
        t_stat, p_val = stats.ttest_1samp(beta_df[col], 0)
        t_test_results[col] = [t_stat, p_val]

    t_test_df = pd.DataFrame.from_dict(
        t_test_results, orient="index", columns=["T-statistic", "P-value"]
    )
    t_test_df.to_csv(os.path.join(save_path, "t_test_results.csv"))

    colors = ["#e87e72", "#56bcc2"]  # Define colors for the boxplot
    beta_df_melt = beta_df.drop(columns=["sub_num"]).melt(
        var_name="columns"
    )  # Melt the dataframe for seaborn boxplot, excluding "sub_num" column
    beta_df_melt["Model"] = np.where(
        beta_df_melt["columns"].str.contains("_rl_"), "rl", "bl"
    )
    # Plot the boxplot
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]  # Use SimHei font
    plt.figure(figsize=figure_size)  # Set the aspect ratio to 2:1
    sns.boxplot(x="columns", y="value", hue="Model", data=beta_df_melt, palette=colors)
    plt.xticks(
        fontsize=15
    )  # Rotate x-axis labels for better visibility and increase font size
    plt.yticks(fontsize=15)  # Increase y-axis font size
    plt.xlabel("模型", fontsize=15)  # Increase x-axis label font size
    plt.ylabel("beta 值", fontsize=15)  # Increase y-axis label font size
    (
        handles,
        labels,
    ) = plt.gca().get_legend_handles_labels()  # Get the current handles and labels
    labels = [
        "强化学习" if label == "rl" else "贝叶斯学习" for label in labels
    ]  # Replace 'rl' with '强化学习' and 'bl' with '贝叶斯学习'
    plt.legend(handles, labels, fontsize=15)  # Set the legend with the new labels
    # Replace x-axis labels according to the instructions
    x_labels = [label.get_text() for label in plt.gca().get_xticklabels()]
    x_labels = ["RL SR" if "rl_sr" in label else label for label in x_labels]
    x_labels = ["RL AB" if "rl_ab" in label else label for label in x_labels]
    x_labels = ["BL SR" if "bl_sr" in label else label for label in x_labels]
    x_labels = ["BL AB" if "bl_ab" in label else label for label in x_labels]
    plt.gca().set_xticklabels(x_labels)
    plt.savefig(os.path.join(save_path, "beta_boxplot.png"))  # Save the plot
    plt.close()
