import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr
import seaborn as sns
import numpy as np


def model_overlap_plot(
    raw_data,
    sub_nums,
    assign_name,
    save_path=None,
    aspect_ratio=1.5,
    print_corr=True,
    show_plot=False,
):
    num_subs = len(sub_nums)  # Number of subjects
    num_models = len(assign_name)  # Number of models

    # Calculate the new figure size based on the given aspect ratio
    new_fig_height = 5 * num_subs
    new_fig_width = new_fig_height * aspect_ratio

    # Create a new figure with subplots
    fig, axes = plt.subplots(
        num_subs, num_models, figsize=(new_fig_width, new_fig_height)
    )

    # Ensure axes is a 2D array even if there's only one row or one column
    if num_subs == 1:
        axes = axes.reshape(1, -1)
    if num_models == 1:
        axes = axes.reshape(-1, 1)

    # Loop through each subject and each model to populate the subplots
    for i, sub_num in enumerate(sub_nums):
        single_sub_data = raw_data[raw_data["sub_num"] == sub_num].reset_index(
            drop=True
        )

        # Extract and preprocess exp_design
        exp_design = single_sub_data["prop"].values
        if "MI" in exp_design:
            exp_design = [0.20 if x == "MI" else x for x in exp_design]
            exp_design = [0.80 if x == "MC" else x for x in exp_design]
        exp_design = [float(x) for x in exp_design]

        # Initialize a dictionary to store model estimates
        model_estimate = {}
        for names in assign_name:
            key = "_".join(names).split("_")[0:2]
            key = "_".join(key)
            model_estimate[key] = single_sub_data[names].mean(axis=1)

        # Plotting and correlation calculation
        colors = ["#e87e72", "#56bcc2"]
        for j, (key, value) in enumerate(model_estimate.items()):
            ax = axes[i, j]  # Select the corresponding subplot
            label = (
                "Bayesian Learning"
                if "bl" in key
                else "Reinforcement Learning"
                if "rl" in key
                else key
            )

            ax.plot(exp_design, color="darkblue", linewidth=2.0, label="exp_design")
            ax.plot(
                abs(1 - value),
                color=colors[j],
                alpha=0.75,
                linewidth=2.0,
                label=label,
            )

            ax.set_title(f"Sub {sub_num}: {label}")
            ax.set_ylim([0, 1])

    plt.tight_layout()

    if print_corr:
        all_exp_design = raw_data["prop"].values
        if "MI" in all_exp_design:
            all_exp_design = [0.20 if x == "MI" else x for x in all_exp_design]
            all_exp_design = [0.80 if x == "MC" else x for x in all_exp_design]
        all_exp_design = [float(x) for x in all_exp_design]
        # Check for NaN values and raise an exception if found
        if np.isnan(exp_design).any() or np.isnan(abs(1 - value)).any():
            raise ValueError(
                "Data contains NaN values, cannot calculate Pearson correlation."
            )

        for i in assign_name:
            # Calculate and print correlation and significance
            # 从 raw_data 中提取对应的所有 subject_num 的 value

            model_data = raw_data[i].mean(axis=1)
            value = np.array(model_data)

            if "bl" in i[0]:
                corr_label = "Bayesian Learning"
            elif "rl" in i[0]:
                corr_label = "Reinforcement Learning"

            # 计算 exp_design 和 values 的相关性
            corr, p_value = pearsonr(all_exp_design, abs(1 - value))
            print(
                f"Correlation between exp_design and {corr_label}: Pearson r = {corr:.2f}, p-value = {p_value:.2e}"
            )

    # Save the figure to the specified path if given
    if save_path is not None:
        plt.savefig(save_path)

    if show_plot:
        plt.show()


# Update the function to include t-test for differences between volatility conditions
def plot_bl_v_boxplot(
    raw_data,
    subject_colname="sub_num",
    volatility_colname="volatile",
    bl_v_colname="bl_sr_v",
    colors=["#e87e72", "#56bcc2"],
    save_path=None,
    fig_size=(10, 6),
    print_ttest=True,
):
    # Convert volatility values to lowercase for consistency
    raw_data[volatility_colname] = raw_data[volatility_colname].str.lower()

    # Group the data by subject and volatility, then calculate the mean for each group
    grouped_data = (
        raw_data.groupby([subject_colname, volatility_colname])[bl_v_colname]
        .mean()
        .reset_index()
    )

    # Check if the required data for t-test is available
    if (
        grouped_data.empty
        or "s" not in grouped_data[volatility_colname].values
        or "v" not in grouped_data[volatility_colname].values
    ):
        raise ValueError("Insufficient data for t-test.")

    # Create the boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=volatility_colname, y=bl_v_colname, data=grouped_data, palette=colors)

    # Additional plot settings
    plt.title("Bayesian Learning")
    plt.xlabel("Volatility")
    plt.ylabel("Subject Mean Value")

    # Perform t-test between s and v conditions
    s_data = grouped_data[grouped_data[volatility_colname] == "s"][bl_v_colname]
    v_data = grouped_data[grouped_data[volatility_colname] == "v"][bl_v_colname]
    t_stat, p_value = ttest_ind(s_data, v_data)

    if print_ttest:
        print(
            f"T-test results between 's' and 'v' conditions: t-statistic = {t_stat:.2f}, p-value = {p_value:.2e}"
        )

    if save_path is not None:
        fig = plt.gcf()
        fig.set_size_inches(fig_size[0], fig_size[1])
        plt.savefig(save_path, dpi=300)


# Re-plot the boxplot for RL model's alpha with all subjects included
def plot_rl_alpha_boxplot(
    alpha_table,
    subject_colname="sub_num",
    alpha_colnames=["alpha_s", "alpha_v"],
    colors=["#e87e72", "#56bcc2"],
    save_path=None,
    fig_size=(10, 6),
):
    # Check if the required columns exist in the dataframe
    for col in [subject_colname] + alpha_colnames:
        if col not in alpha_table.columns:
            print(f"Error: Missing required column '{col}' in the dataframe.")
            return

    # Check for missing values
    if alpha_table.isnull().any().any():
        print(
            "Warning: The dataframe contains missing values."
            "Please handle them before proceeding."
        )
        return

    # Reshape the data for boxplot
    melted_data = alpha_table.melt(
        id_vars=subject_colname,
        value_vars=alpha_colnames,
        var_name="volatile",
        value_name="alpha",
    )
    melted_data["volatile"] = melted_data["volatile"].map(
        {"alpha_s": "s", "alpha_v": "v"}
    )

    # Create the boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="volatile", y="alpha", data=melted_data, palette=colors)

    # Additional plot settings
    plt.title("Boxplot of RL Alpha")
    plt.xlabel("Volatility Condition")
    plt.ylabel("Alpha Value")

    # Perform t-test between s and v conditions
    s_data = alpha_table[next(col for col in alpha_colnames if "s" in col)]
    v_data = alpha_table[next(col for col in alpha_colnames if "v" in col)]
    t_stat, p_value = ttest_ind(s_data, v_data)

    print(
        f"T-test results between 's' and 'v' conditions: t-statistic = {t_stat:.2f}, p-value = {p_value:.2e}"
    )

    if save_path is not None:
        fig = plt.gcf()
        fig.set_size_inches(fig_size[0], fig_size[1])
        plt.savefig(save_path, dpi=300)
