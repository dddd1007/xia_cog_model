## 编写各强化模型

#= 
模型分别为
- ab
- ab_volatility
- sr
- sr_volatility
- sr_decay
- sr_volatility_decay

概念解释
- ab : 抽象的刺激反应联结概念
- sr : 具体的刺激反应联结概念
- decay : 未激活的联结概念的概率衰减
- volatilty : 考虑环境不稳定性分别估计学习率

输入数据编码
- stim_consistency_seq:
    + 0: 不一致
    + 1: 一致
- stim_loc_seq:
    + 0: 左
    + 1: 右
- reaction_loc_seq:
    + 0: 左
    + 1: 右
- exp_volatility_seq:
    + 0: 稳定 (stable)
    + 1: 不稳定 (volatility)
=#

function ab_model(α::Float64, stim_consistency_seq::Vector{Int}; drop_last_one=true)
    # Init predict code sequence
    predict_seq = Vector{Float64}(undef, (length(stim_consistency_seq) + 1))
    prediction_error_seq = Vector{Float64}(undef, length(stim_consistency_seq))
    predict_seq[1] = 0.5

    # Update predict code sequence
    for i in 1:length(stim_consistency_seq)
        PE = stim_consistency_seq[i] - predict_seq[i]
        predict_seq[i + 1] = predict_seq[i] + α * PE
        prediction_error_seq[i] = PE
    end

    # Return the result
    if drop_last_one
        return Dict("Predicted sequence" => predict_seq[1:(length(predict_seq) - 1)],
                    "Prediciton error" => prediction_error_seq)
    else
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq)
    end
end

function ab_volatility_model(α_s::Float64, α_v::Float64, stim_consistency_seq::Vector{Int},
                             exp_volatility_seq::Vector{Int}; drop_last_one=true)
    # Init predict code sequence
    predict_seq = Vector{Float64}(undef, (length(stim_consistency_seq) + 1))
    prediction_error_seq = Vector{Float64}(undef, length(stim_consistency_seq))
    predict_seq[1] = 0.5

    # Update predict code sequence
    for i in 1:length(stim_consistency_seq)
        PE = stim_consistency_seq[i] - predict_seq[i]
        if exp_volatility_seq[i] == 0
            predict_seq[i + 1] = predict_seq[i] + α_s * PE
        else
            predict_seq[i + 1] = predict_seq[i] + α_v * PE
        end
        prediction_error_seq[i] = PE
    end

    # Return the result
    if drop_last_one
        return Dict("Predicted sequence" => predict_seq[1:(length(predict_seq) - 1)],
                    "Prediciton error" => prediction_error_seq)
    else
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq)
    end
end

# 使用新的左右分开估计的模型来替代该模型
# function sr_model(α::Float64, stim_loc_seq::Vector{Int}, reaction_loc_seq::Vector{Int};
#                   drop_last_one=true)
#     # Init predict code sequence
#     predict_seq = Vector{Float64}(undef, length(stim_loc_seq))
#     prediction_error_seq = Vector{Float64}(undef, length(stim_loc_seq))
#     predict_l_l = 0.5
#     predict_r_l = 0.5

#     # Update predict code sequence
#     for i in 1:length(stim_loc_seq)
#         if stim_loc_seq[i] == 0 # Stim come from left
#             predict_seq[i] = predict_l_l
#             PE = reaction_loc_seq[i] - predict_l_l
#             predict_l_l = predict_l_l + α * PE
#             prediction_error_seq[i] = PE
#         else
#             predict_seq[i] == predict_r_l
#             PE = reaction_loc_seq[i] - predict_r_l
#             predict_r_l = predict_r_l + α * PE
#             prediction_error_seq[i] = PE
#         end
#     end

#     # Return the result
#     if drop_last_one
#         return Dict("Predicted sequence" => predict_seq,
#                     "Prediciton error" => prediction_error_seq)
#     else
#         return Dict("Predicted sequence" => predict_seq,
#                     "Prediciton error" => prediction_error_seq)
#     end
# end

function sr_volatility_model(α_s::Float64, α_v::Float64, stim_loc_seq,
                             reaction_loc_seq, exp_volatility_seq;
                             drop_last_one=true)
    # Init predict code sequence
    predict_seq          = Vector{Float64}(undef, length(stim_loc_seq))
    predict_seq_l        = [0.5]
    predict_seq_r        = [0.5]
    prediction_error_seq = Vector{Float64}(undef, length(stim_loc_seq))

    predict_seq[1]       = 0.5
    predict_l_l          = 0.5
    predict_r_l          = 0.5

    # Update predict code sequence
    for i in 1:length(stim_loc_seq)
        if exp_volatility_seq[i] == 0
            if stim_loc_seq[i] == 0 # Stim come from left
                predict_seq[i] = predict_l_l
                PE = reaction_loc_seq[i] - predict_l_l
                push!(predict_seq_l, predict_l_l + α_s * PE)
                predict_l_l = predict_l_l + α_s * PE
                prediction_error_seq[i] = PE
            else
                predict_seq[i] = predict_r_l
                PE = reaction_loc_seq[i] - 1 - predict_r_l
                push!(predict_seq_r, predict_r_l + α_s * PE)
                predict_r_l = predict_r_l + α_s * PE
                prediction_error_seq[i] = PE
            end
        else
            if stim_loc_seq[i] == 0 # Stim come from left
                predict_seq[i] = predict_l_l
                PE = reaction_loc_seq[i] - predict_l_l
                push!(predict_seq_l, predict_l_l + α_v * PE)
                predict_l_l = predict_l_l + α_v * PE
                prediction_error_seq[i] = PE
            else
                predict_seq[i] = predict_r_l
                PE = reaction_loc_seq[i] - predict_r_l
                push!(predict_seq_r, predict_r_l + α_v * PE)
                predict_r_l = predict_r_l + α_v * PE
                prediction_error_seq[i] = PE
            end
        end
    end

    # Return the result
    if drop_last_one
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l[1:(length(predict_seq_l) - 1)],
                    "Predicied Right sequence" => predict_seq_r[1:(length(predict_seq_r) - 1)])
    else
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l,
                    "Predicied Right sequence" => predict_seq_r)
    end
end

function sr_sep_alpha_model(α_l::Float64, α_r::Float64, stim_loc_seq::Vector{Int},
                            reaction_loc_seq::Vector{Int}; drop_last_one=true)
    # Init predict code sequence
    predict_seq = Vector{Float64}(undef, length(stim_loc_seq))
    prediction_error_seq = Vector{Float64}(undef, length(stim_loc_seq))
    predict_seq_l = [0.5]
    predict_seq_r = [0.5]
    predict_l_l = 0.5
    predict_r_l = 0.5
    predict_seq[1] = 0.5

    # Update predict code sequence
    for i in 1:length(stim_loc_seq)
        if stim_loc_seq[i] == 0 # Stim come from left
            PE = reaction_loc_seq[i] - predict_l_l
            predict_seq[i] = predict_l_l
            predict_l_l = predict_l_l + α_l * PE
            push!(predict_seq_l, predict_l_l + α_l * PE)
            push!(predict_seq_r, predict_r_l)
            prediction_error_seq[i] = PE
        else
            PE = reaction_loc_seq[i] - predict_r_l
            predict_seq[i] = predict_r_l
            predict_r_l = predict_r_l + α_r * PE
            push!(predict_seq_l, predict_l_l)
            push!(predict_seq_r, predict_r_l + α_r * PE)
            prediction_error_seq[i] = PE
        end
    end

    # Return the result
    if drop_last_one
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l[1:(length(predict_seq_l) - 1)],
                    "Predicied Right sequence" => predict_seq_r[1:(length(predict_seq_r) - 1)])
    else
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l,
                    "Predicied Right sequence" => predict_seq_r)
    end
end

function sr_sep_alpha_volatility_model(α_s_l::Float64, α_s_r::Float64, 
                                       α_v_l::Float64, α_v_r::Float64, 
                                       stim_loc_seq::Vector{Int},
                                       reaction_loc_seq::Vector{Int},
                                       exp_volatility_seq::Vector{Int}; drop_last_one=true)
    # Init predict code sequence
    predict_seq = Vector{Float64}(undef, length(stim_loc_seq))
    prediction_error_seq = Vector{Float64}(undef, length(stim_loc_seq))
    predict_seq_l = [0.5]
    predict_seq_r = [0.5]
    predict_l_l = 0.5
    predict_r_l = 0.5
    predict_seq[1] = 0.5

    # Update predict code sequence
    for i in 1:length(stim_loc_seq)
        if exp_volatility_seq[i] == 0
            if stim_loc_seq[i] == 0 # Stim come from left
                predict_seq[i] = predict_l_l
                PE = reaction_loc_seq[i] - predict_l_l
                predict_l_l = predict_l_l + α_s_l * PE
                push!(predict_seq_l, predict_l_l + α_s_l * PE)
                push!(predict_seq_r, predict_r_l)
                prediction_error_seq[i] = PE
            else
                predict_seq[i] = predict_r_l
                PE = reaction_loc_seq[i] - predict_r_l
                predict_r_l = predict_r_l + α_s_r * PE
                push!(predict_seq_l, predict_l_l)
                push!(predict_seq_r, predict_r_l + α_s_r * PE)
                prediction_error_seq[i] = PE
            end
        else
            if stim_loc_seq[i] == 0 # Stim come from left
                predict_seq[i] = predict_l_l
                PE = reaction_loc_seq[i] - predict_l_l
                predict_l_l = predict_l_l + α_v_l * PE
                push!(predict_seq_l, predict_l_l + α_v_l * PE)
                push!(predict_seq_r, predict_r_l)
                prediction_error_seq[i] = PE
            else
                predict_seq[i] = predict_r_l
                PE = reaction_loc_seq[i] - predict_r_l
                predict_r_l = predict_r_l + α_v_r * PE
                push!(predict_seq_l, predict_l_l)
                push!(predict_seq_r, predict_r_l + α_v_r * PE)
                prediction_error_seq[i] = PE
            end
        end
    end

    # Return the result
    if drop_last_one
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l[1:(length(predict_seq_l) - 1)],
                    "Predicied Right sequence" => predict_seq_r[1:(length(predict_seq_r) - 1)])
    else
        return Dict("Predicted sequence" => predict_seq,
                    "Prediciton error" => prediction_error_seq,
                    "Predicied Left sequence" => predict_seq_l,
                    "Predicied Right sequence" => predict_seq_r)
    end
end