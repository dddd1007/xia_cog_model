import matplotlib.pyplot as plt


def overlap_plot(raw_data, sub_num, assign_name, ax=None):
    single_sub_data = raw_data[raw_data["sub_num"] == sub_num].reset_index()

    # 提取实验设计
    exp_design = single_sub_data["prop"].values
    # 检查 exp_design 中是否包含 "MI"
    if "MI" in exp_design:
        # 如果包含 "MI"，则进行替换
        exp_design = [0.20 if x == "MI" else x for x in exp_design]
        exp_design = [0.80 if x == "MC" else x for x in exp_design]
        # 转换 exp_design 为 int list
        exp_design = [float(x) for x in exp_design]
    else:
        # 如果不包含 "MI"，则确认 exp_design 是否是一个 int 的 list
        if not all(isinstance(x, float) for x in exp_design):
            # 如果不是，转换成 int 的 list
            exp_design = [float(x) for x in exp_design]

    # 根据变量提取 LL+RR 的数据
    model_estimate = {}  # 初始化一个空字典用于存储模型估计值
    for names in assign_name:  # 遍历 assign_name 中的每个元素
        key = "_".join(names).split("_")[
            0:2
        ]  # 将 names 中的元素连接成字符串，然后按 "_" 分割，取前两个元素
        key = "_".join(key)  # 将取出的两个元素再连接成字符串，作为字典的键
        model_estimate[key] = single_sub_data[names].mean(axis=1)

    # 如果 ax 参数为 None，创建一个新的 figure 和 axes
    if ax is None:
        fig, ax = plt.subplots(
            1,
            len(model_estimate),
            figsize=(5 * len(model_estimate), 1 * len(model_estimate)),
        )

    # 设置颜色列表
    colors = ["#e87e72", "#56bcc2"]

    # 遍历 model_estimate 中的每个键值对
    for i, (key, value) in enumerate(model_estimate.items()):
        if "bl" in key:
            key = "Bayesian Learning"
        elif "rl" in key:
            key = "Reinforcement Learning"

        # 绘制 exp_design 的线，增加 linewidth 参数使线条更粗
        ax[i].plot(
            exp_design, color="darkblue", linewidth=2.0, label="exp_design"
        )
        # 绘制 model_estimate 的线，增加 linewidth 参数使线条更粗
        ax[i].plot(
            abs(1 - value),
            color=colors[i],
            alpha=0.75,
            linewidth=2.0,
            label=key,
        )
        # 设置子图的 title
        ax[i].set_title(key)
        # 设置子图的纵轴范围
        ax[i].set_ylim([0, 1])

    # 如果 ax 参数为 None，显示图像
    if ax is None:
        plt.show()


# def overlap_plot(raw_data, sub_num, assign_name):
#     single_sub_data = raw_data[raw_data["sub_num"] == sub_num]

#     # 提取实验设计
#     exp_design = single_sub_data["prop"].values
#     # 检查 exp_design 中是否包含 "MI"
#     if "MI" in exp_design:
#         # 如果包含 "MI"，则进行替换
#         exp_design = [0.20 if x == "MI" else x for x in exp_design]
#         exp_design = [0.80 if x == "MC" else x for x in exp_design]
#         # 转换 exp_design 为 int list
#         exp_design = [float(x) for x in exp_design]
#     else:
#         # 如果不包含 "MI"，则确认 exp_design 是否是一个 int 的 list
#         if not all(isinstance(x, float) for x in exp_design):
#             # 如果不是，转换成 int 的 list
#             exp_design = [float(x) for x in exp_design]

#     # 根据变量提取 LL+RR 的数据
#     model_estimate = {}  # 初始化一个空字典用于存储模型估计值
#     for names in assign_name:  # 遍历 assign_name 中的每个元素
#         key = "_".join(names).split("_")[
#             0:2
#         ]  # 将 names 中的元素连接成字符串，然后按 "_" 分割，取前两个元素
#         key = "_".join(key)  # 将取出的两个元素再连接成字符串，作为字典的键
#         model_estimate[key] = single_sub_data[names].mean(axis=1)

#     # 创建一个新的 figure
#     fig, axs = plt.subplots(
#         1,
#         len(model_estimate),
#         figsize=(5 * len(model_estimate), 1 * len(model_estimate)),
#     )

#     # 设置颜色列表
#     colors = ["#e87e72", "#56bcc2"]

#     # 遍历 model_estimate 中的每个键值对
#     for i, (key, value) in enumerate(model_estimate.items()):
#         if "bl" in key:
#             key = "Bayesian Learning"
#         elif "rl" in key:
#             key = "Reinforcement Learning"

#         # 绘制 exp_design 的线，增加 linewidth 参数使线条更粗
#         axs[i].plot(
#             exp_design, color="darkblue", linewidth=2.0, label="exp_design"
#         )
#         # 绘制 model_estimate 的线，增加 linewidth 参数使线条更粗
#         axs[i].plot(
#             abs(1 - value),
#             color=colors[i],
#             alpha=0.75,
#             linewidth=2.0,
#             label=key,
#         )
#         # 设置子图的 title
#         axs[i].set_title(key)
#         # 设置子图的纵轴范围
#         axs[i].set_ylim([0, 1])

#     # 显示图像
#     plt.show()
