using Test, GLM, StatsBase
import CSV

using Models, DataFramesMeta

# 生成测试数据
begin
    all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv");
    begin
        color_rule = Dict("red" => "0" , "green" => "1")
        congruency_rule = Dict("con" => "1", "inc" => "0")
        Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
        loc_rule = Dict("left" => "0", "right" => "1")
        transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule, 
            "stim_loc" => loc_rule, "congruency" => congruency_rule)
    end
    transform_data!(all_data, transform_rule)
    sub1_data = @where(all_data, :Subject .== "sub01_Yangmiao");
    begin
        env_idx_dict = Dict("stim_task_related" => "stim_color", 
                            "stim_task_unrelated" => "stim_loc", 
                            "stim_action_congruency" => "congruency", 
                            "correct_action" => "correct_action",
                            "env_type" => "condition", "sub_tag" => "Subject")
        sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
                            "corrections" => "Type", "sub_tag" => "Subject")
    end
    sub1_env, sub1_subinfo = init_env_sub(sub1_data, env_idx_dict, sub_idx_dict);
end

# 测试函数计算的基本要素


# 冲突水平的计算
test1 = [0.6;0.4;0.7;0.3]
@test Models.RLModels.calc_CCC(test1, (0,0)) ≈ 0.2

test2 = [0.6, 0.4]
@test Models.RLModels.calc_CCC(test2, 0) ≈ 0.2

# 价值矩阵更新
weight_matrix = zeros(Float64, (2, 4))
weight_matrix[1,:] = [0.5,0.5,0.5,0.5]
weight_matrix[2,:] = Models.RLModels.update_options_weight_matrix(weight_matrix[1,:] , 0.5, 0.9, (0,0))
@test weight_matrix[2,:] == [0.75, 0.25, 0.5, 0.5]



# 测试拟合带SoftMax的计算过程

@testset "RLModels_with_Softmax" begin
    
    # 简单模式下学习
    begin
        α_v = 0.1
        β_v = 5
        α_s = 0.2
        β_s = 10
        decay = 0.9
    end

    agent = RLModels_SoftMax.RLLearner_basic(α_v, β_v, α_s, β_s, decay)
    @test RLModels_SoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo)

    # 错误试次下学习率不同
    begin
        α_v = 0.1
        β_v = 5
        α_s = 0.2
        β_s = 10

        α_v_error = 0.1
        β_v_error = 5
        α_s_error = 0.2
        β_s_error = 10

        decay = 0.9
    end

    agent = RLModels_SoftMax.RLLearner_witherror(α_v, β_v, α_s, β_s, α_v_error, β_v_error, α_s_error, β_s_error, decay)
    @test RLModels_SoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo)

    # 有主动控制的CCC的模型
    begin
        α_v = 0.1
        β_v = 5
        α_s = 0.2
        β_s = 10

        α_v_error = 0.1
        β_v_error = 5
        α_s_error = 0.2
        β_s_error = 10

        α_v_CCC = 0.1
        β_v_CCC = 5
        α_s_CCC = 0.2
        β_s_CCC = 10

        decay = 0.9
    end

    agent = RLModels_SoftMax.RLLearner_withCCC(α_v, β_v, α_s, β_s, α_v_error, β_v_error, α_s_error, β_s_error, α_v_CCC, β_v_CCC, α_s_CCC, β_s_CCC, decay)
    RLModels_SoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo)
end

# 简单模式下学习
begin
    α_v = 0.1
    α_s = 0.2
end

agent = RLModels_no_SoftMax.RLLearner_basic(α_v, α_s, decay)
result = RLModels_no_SoftMax.rl_learning_ab(sub1_env, agent, sub1_subinfo)

begin
    α_v = 0.1
    β_v = 5
    α_s = 0.2
    β_s = 10

    α_v_error = 0.1
    β_v_error = 5
    α_s_error = 0.2
    β_s_error = 10

    decay = 0.9
end

agent = RLModels_SoftMax.RLLearner_witherror(α_v, β_v, α_s, β_s, α_v_error, β_v_error, α_s_error, β_s_error, decay)
result = RLModels_SoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo)

begin
    α_v = 0.1
    β_v = 5.0
    α_s = 0.2
    β_s = 10

    α_v_error = 0.1
    β_v_error = 5.0
    α_s_error = 0.2
    β_s_error = 10

    α_v_CCC = 0.1
    β_v_CCC = 5.0
    α_s_CCC = 0.2
    β_s_CCC = 10
    
    CCC = 0.5
    decay = 0.9
end
