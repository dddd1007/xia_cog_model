using .RLModels
using Hyperopt, RecursiveArrayTools, StatsBase, DataFrames, GLM
using CategoricalArrays

#####
##### 工具性函数
#####

# 定义评估变量关系的函数
function evaluate_relation(x::AbstractArray, y::AbstractArray)
    data = DataFrame(; x=x, y=y)
    reg_result = lm(@formula(y ~ x), data)
    β_value = coef(reg_result)[2]
    aic_value = aic(reg_result)
    bic_value = bic(reg_result)
    r2_value = r2(reg_result)
    mse_value = deviance(reg_result) / dof_residual(reg_result)
    loglikelihood_value = loglikelihood(reg_result)
    result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value,
                  :MSE => mse_value, :Loglikelihood => loglikelihood_value)
    return result
end

function evaluate_relation(dataframe::DataFrame, formula::FormulaTerm)
    reg_result = lm(formula, dataframe)
    β_value = coef(reg_result)[2]
    aic_value = aic(reg_result)
    bic_value = bic(reg_result)
    r2_value = r2(reg_result)
    mse_value = deviance(reg_result) / dof_residual(reg_result)
    loglikelihood_value = loglikelihood(reg_result)
    result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value,
                  :MSE => mse_value, :Loglikelihood => loglikelihood_value)
    return result
end

# 根据最优参数重新拟合模型
function model_recovery(env::ExpEnv, realsub::RealSub, opt_params; model_type)
    if model_type == :_1a
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α], opt_params[:α], 0)
    elseif model_type == :_1a1d
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α], opt_params[:α],
                                                   opt_params[:decay])
    elseif model_type == :_1a1d1e
        agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params[:α], opt_params[:α],
                                                       opt_params[:α_error],
                                                       opt_params[:α_error],
                                                       opt_params[:decay])
    elseif model_type == :_1a1d1e1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[:α], opt_params[:α],
                                                     opt_params[:α_error],
                                                     opt_params[:α_error],
                                                     opt_params[:α_CCC], opt_params[:α_CCC],
                                                     opt_params[:CCC], opt_params[:decay])
    elseif model_type == :_1a1d1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(opt_params[:α],
                                                              opt_params[:α],
                                                              opt_params[:α_CCC],
                                                              opt_params[:α_CCC],
                                                              opt_params[:CCC],
                                                              opt_params[:decay])
    elseif model_type == :_2a
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α_v], opt_params[:α_s], 0)
    elseif model_type == :_2a1d
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α_v], opt_params[:α_s],
                                                   opt_params[:decay])
    elseif model_type == :_2a1d1e
        agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params[:α_v], opt_params[:α_s],
                                                       opt_params[:α_error],
                                                       opt_params[:α_error],
                                                       opt_params[:decay])
    elseif model_type == :_2a1d1e1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[:α_v], opt_params[:α_s],
                                                     opt_params[:α_error],
                                                     opt_params[:α_error],
                                                     opt_params[:α_CCC], opt_params[:α_CCC],
                                                     opt_params[:CCC], opt_params[:decay])
    elseif model_type == :_2a1d1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(opt_params[:α_v],
                                                              opt_params[:α_s],
                                                              opt_params[:α_CCC],
                                                              opt_params[:α_CCC],
                                                              opt_params[:CCC],
                                                              opt_params[:decay])
    end

    if model_type == :_1a || model_type == :_2a
        return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub; dodecay=false)
    else
        return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
    end
end

#####
##### 强化学习模型的模型拟合
#####

# Type1 不进行 detrend
function fit_RL_base(env, realsub, looptime; model_type)

    ## Fit the hyperparameters
    if model_type == :_1a
        ho = @hyperopt for i = looptime,

                           α = [0.001:0.001:0.999;]
            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay=false)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:0.999), decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e
        ho = @hyperopt for i = looptime, α = LinRange(0.001:0.001:0.999),

                           α_error = LinRange(0.001:0.001:1), decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_witherror(α, α, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e1CCC
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:1),
                           α_error = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_withCCC(α, α, α_error, α_error, α_CCC,
                                                         α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1CCC
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α, α, α_CCC, α_CCC, CCC,
                                                                  decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1), α_s = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub;
                                                           dodecay=false)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e
        ho = @hyperopt for i = looptime,

                            α_v = LinRange(0.001:0.001:1),
                            α_s = LinRange(0.001:0.001:1),
                            α_error = LinRange(0.001:0.001:1),
                            decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_error, α_error,
                                                           decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e1CCC
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           α_error = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_CCC,
                                                         α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1CCC
        ho = @hyperopt for i = looptime,

                            α_v = LinRange(0.001:0.001:1),
                            α_s = LinRange(0.001:0.001:1),
                            α_CCC = LinRange(0.001:0.001:1),
                            CCC = LinRange(-0.001:-0.001:-1),
                            decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α_v, α_s, α_CCC, α_CCC,
                                                                  CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    end
    optim_params_value, eval_result = ho.minimizer, ho.minimum
    optim_params = Dict(zip(ho.params, optim_params_value))
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[!, :MSE] = ho.results

    return (optim_params, eval_result, verbose_table)
end

# Type2 按照每次改变比例算一个 miniblock 进行 detrend
function fit_RL_detrend_miniblock(env, realsub, looptime; model_type)
    ## Add miniblock
    cache_prop = realsub.prop_seq[1]
    cache_index = 1
    prop_seq_changed = Array{Int64,1}(undef, length(realsub.prop_seq))
    prop_seq_changed[1] = cache_index
    for i = 2:length(realsub.prop_seq)
        if realsub.prop_seq[i] != cache_prop
            cache_index += 1
            cache_prop = realsub.prop_seq[i]
        end
        prop_seq_changed[i] = cache_index
    end
    ## Fit the hyperparameters
    if model_type == :_1a
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:0.999)
            agent = RLModels.NoSoftMax.RLLearner_basic(α, α, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub;
                                                           dodecay=false)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:0.999),
                           decay = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1e
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:0.999),
                           α_error = LinRange(0.001:0.001:1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_witherror(α, α, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1e1CCC
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:1),
                           α_error = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α, α, α_error, α_error, α_CCC,
                                                         α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1CCC
        ho = @hyperopt for i = looptime,

                           α = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α, α, α_CCC, α_CCC, CCC,
                                                                  decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1), α_s = LinRange(0.001:0.001:1)
            agent = RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub;
                                                           dodecay=false)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1e
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           α_error = LinRange(0.001:0.001:1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_error, α_error,
                                                           decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1e1CCC
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           α_error = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_CCC,
                                                         α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1CCC
        ho = @hyperopt for i = looptime,

                           α_v = LinRange(0.001:0.001:1),
                           α_s = LinRange(0.001:0.001:1),
                           α_CCC = LinRange(0.001:0.001:1),
                           CCC = LinRange(-0.001:-0.001:-1),
                           decay = LinRange(0.001:0.001:1)

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α_v, α_s, α_CCC, α_CCC,
                                                                  CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(;
                                             predicted_var=model_stim[:p_selection_history],
                                             RT=realsub.RT,
                                             miniblock=CategoricalArray(prop_seq_changed;
                                                                        ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    end

    optim_params_value, eval_result = ho.minimizer, ho.minimum
    optim_params = Dict(zip(ho.params, optim_params_value))
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[!, :MSE] = ho.results

    return (optim_params, eval_result, verbose_table)
end

function correct_AIC(model_type, AIC)
    if model_type == :_1a
        params_num = 1
    elseif model_type == :_1a1d
        params_num = 2
    elseif model_type == :_1a1d1e
        params_num = 3
    elseif model_type == :_1a1d1e1CCC
        params_num = 4
    elseif model_type == :_1a1d1CCC
        params_num = 3
    elseif model_type == :_2a
        params_num = 2
    elseif model_type == :_2a1d
        params_num = 3
    elseif model_type == :_2a1d1e
        params_num = 4
    elseif model_type == :_2a1d1e1CCC
        params_num = 5
    elseif model_type == :_2a1d1CCC
        params_num = 4
    end

    corrected_AIC = AIC + 2 * params_num - 2
end

function correct_AIC(model_type, AIC, miniblock_count)
    if model_type == :_1a
        params_num = 1
    elseif model_type == :_1a1d
        params_num = 2
    elseif model_type == :_1a1d1e
        params_num = 3
    elseif model_type == :_1a1d1e1CCC
        params_num = 4
    elseif model_type == :_1a1d1CCC
        params_num = 3
    elseif model_type == :_2a
        params_num = 2
    elseif model_type == :_2a1d
        params_num = 3
    elseif model_type == :_2a1d1e
        params_num = 4
    elseif model_type == :_2a1d1e1CCC
        params_num = 5
    elseif model_type == :_2a1d1CCC
        params_num = 4
    end

    corrected_AIC = AIC + 2 * params_num - 2 - 2 * miniblock_count
end

#####
##### 整合函数进行估计
#####

function fit_and_evaluate_base(env, realsub; model_type, number_iterations)
    optim_param, _, _ = fit_RL_base(env, realsub, number_iterations, model_type=model_type)
    p_history = model_recovery(env, realsub, optim_param, model_type=model_type)[:p_selection_history]
    eval_result = evaluate_relation(p_history, realsub.RT)
    corrected_AIC = correct_AIC(model_type, eval_result[:AIC])

    return Dict(:optim_param => optim_param, :p_history => p_history,
                :AIC => corrected_AIC, :MSE => eval_result[:MSE], :model_type => model_type)
end

function fit_and_evaluate_miniblock(env, realsub; model_type, number_iterations)
    optim_param, _, _ = fit_RL_detrend_miniblock(env, realsub, number_iterations, model_type=model_type)
    p_history = model_recovery(env, realsub, optim_param, model_type=model_type)[:p_selection_history]

    ## Add miniblock
    cache_prop = realsub.prop_seq[1]
    cache_index = 1
    prop_seq_changed = Array{Int64,1}(undef, length(realsub.prop_seq))
    prop_seq_changed[1] = cache_index
    for i = 2:length(realsub.prop_seq)
        if realsub.prop_seq[i] != cache_prop
            cache_index += 1
            cache_prop = realsub.prop_seq[i]
        end
        prop_seq_changed[i] = cache_index
    end

    validation_dataframe = DataFrame(predicted_var=p_history, RT=realsub.RT,
                                     miniblock=CategoricalArray(prop_seq_changed,
                                                                ordered=false))
    validation_formula = @formula(RT ~ predicted_var + miniblock)
    eval_result = evaluate_relation(validation_dataframe, validation_formula)

    miniblock_count = length(unique(prop_seq_changed))
    corrected_AIC = correct_AIC(model_type, eval_result[:AIC], miniblock_count)

    return Dict(:optim_param => optim_param, :p_history => p_history,
                :AIC => corrected_AIC, :MSE => eval_result[:MSE],
                :model_type => model_type)
end
