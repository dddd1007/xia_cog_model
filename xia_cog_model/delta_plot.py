import pandas as pd


def calc_simon_effect(data, n_bins, congruency_col, inc_value, factors):
    # 获取 congruency_col 中不是 inc_value 的另一个值，如果有多于两个值则报错
    congruency_values = data[congruency_col].unique()
    if len(congruency_values) != 2:
        raise ValueError("Congruency column must contain exactly two unique values.")
    con_value = [v for v in congruency_values if v != inc_value][0]

    # 对每个被试的数据先按照 factors 和 congruency_col 进行分组
    # 按照从小到大进行排序，然后平均分为 n_bins 个 bin
    # 每个 bin 的每个条件下计算平均值
    bins_data = []
    for name, group in data.groupby(["subject_num"] + factors):
        group = group.sort_values("key_resp.rt")
        group["bin"] = pd.cut(
            group["key_resp.rt"], bins=n_bins, labels=range(1, n_bins + 1)
        )
        group_mean = group.groupby(["bin", congruency_col]).mean().reset_index()
        group_mean["subject_num"] = name[0]
        for factor, value in zip(factors, name[1:]):
            group_mean[factor] = value
        bins_data.append(group_mean)

    bins_data = pd.concat(bins_data)

    # 在每个 bin 内计算每个 factors 下的
    # inc_value 的平均值 - con_value 的平均值的数据得到 Simon 效应
    simon_effect_data = []
    for name, group in bins_data.groupby(["subject_num"] + factors + ["bin"]):
        inc_mean_value = group[group[congruency_col] == inc_value][
            "key_resp.rt"
        ].values[0]
        con_mean_value = group[group[congruency_col] == con_value][
            "key_resp.rt"
        ].values[0]
        simon_effect = inc_mean_value - con_mean_value
        simon_effect_data.append(
            {
                "subject_num": name[0],
                **{factor: value for factor, value in zip(factors, name[1:-1])},
                "bin": name[-1],
                "simon_effect": simon_effect,
            }
        )

    simon_effect_data = pd.DataFrame(simon_effect_data)

    return simon_effect_data
