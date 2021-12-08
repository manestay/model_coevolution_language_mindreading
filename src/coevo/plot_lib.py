import cPickle as pickle

import numpy as np
from scipy import stats

import context
import lex
import prior
import pop

copy_specification = ''  # Can be set to e.g. '_c1' or simply to '' if there is only one copy


def calc_mean_and_conf_invs_distribution(n_runs, n_copies, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, cut_off_point, selected_hyps_new_lex_order_all_runs):
    hist_values_averaged_per_run = np.zeros((n_runs, 3))
    for r in range(n_runs):
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_all_runs[r][cut_off_point:n_iterations]
        selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_final_section.flatten()
        if which_hyps_on_graph == 'lex_hyps_only' or which_hyps_on_graph == 'lex_hyps_collapsed':
            n_lex_hyps = len(lexicon_hyps)
            for i in range(len(selected_hyps_new_lex_order_final_section)):
                hyp_index = selected_hyps_new_lex_order_final_section[i]
                if hyp_index > (n_lex_hyps - 1):
                    selected_hyps_new_lex_order_final_section[i] = hyp_index - n_lex_hyps
        hist_values = np.histogram(selected_hyps_new_lex_order_final_section, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
        hist_values_averaged = np.divide(hist_values[0].astype(float), [float(len(min_info_indices)), float(len(intermediate_info_indices)), float(len(max_info_indices))])
        hist_values_averaged_per_run[r] = hist_values_averaged
    hist_values_averaged_per_run_optimal_first = np.fliplr(hist_values_averaged_per_run)
    # hist_values_norm = np.divide(hist_values_averaged_per_run_optimal_first, np.sum(hist_values_averaged_per_run_optimal_first, axis=1)[:, None])
    mean_selected_hyps_by_lex_type = np.mean(hist_values_averaged_per_run_optimal_first, axis=0)
    std_selected_hyps_by_lex_type = np.std(hist_values_averaged_per_run_optimal_first, axis=0)
    conf_invs_selected_hyps_by_lex_type = stats.norm.interval(0.95, loc=mean_selected_hyps_by_lex_type, scale=std_selected_hyps_by_lex_type / np.sqrt(n_runs*n_copies))
    conf_invs_selected_hyps_by_lex_type = np.array(conf_invs_selected_hyps_by_lex_type)
    lower_yerr_selected_hyps_for_plot = np.subtract(mean_selected_hyps_by_lex_type, conf_invs_selected_hyps_by_lex_type[0])
    upper_yerr_selected_hyps_for_plot = np.subtract(conf_invs_selected_hyps_by_lex_type[1], mean_selected_hyps_by_lex_type)
    yerr_selected_hyps_for_plot = np.array([lower_yerr_selected_hyps_for_plot, upper_yerr_selected_hyps_for_plot])
    hypothesis_count_proportions = np.divide(mean_selected_hyps_by_lex_type, np.sum(mean_selected_hyps_by_lex_type))
    yerr_scaled_selected_hyps_for_plot = np.divide(yerr_selected_hyps_for_plot, np.sum(mean_selected_hyps_by_lex_type))

    return hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot


def calc_mean_and_conf_invs_distribution2(n_runs, n_copies, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, selected_hyps_new_lex_order_all_runs):
    n_iters = len(selected_hyps_new_lex_order_all_runs[0])

    hist_values_averaged_per_run = np.zeros((n_runs, 3))
    iter_prop = [None] * n_iters
    for r in range(n_runs):
        curr_run_hyps = selected_hyps_new_lex_order_all_runs[r]
        for i in range(n_iters):
            curr_iter_hyps = curr_run_hyps[i]
            hist_values, _ = np.histogram(curr_iter_hyps, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
            if iter_prop[i] is None:
                iter_prop[i] = hist_values
            else:
                iter_prop[i] += hist_values
    for i, iter_stats in enumerate(iter_prop):
        props = np.true_divide(iter_stats,sum(iter_stats))
        print('run {} proportions are {}'.format(i, props))

    #     selected_hyps_new_lex_order_final_section = selected_hyps_new_lex_order_final_section.flatten()
    #     if which_hyps_on_graph == 'lex_hyps_only' or which_hyps_on_graph == 'lex_hyps_collapsed':
    #         n_lex_hyps = len(lexicon_hyps)
    #         for i in range(len(selected_hyps_new_lex_order_final_section)):
    #             hyp_index = selected_hyps_new_lex_order_final_section[i]
    #             if hyp_index > (n_lex_hyps - 1):
    #                 selected_hyps_new_lex_order_final_section[i] = hyp_index - n_lex_hyps
    #     hist_values = np.histogram(selected_hyps_new_lex_order_final_section, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
    #     import pdb; pdb.set_trace()
    #     hist_values_averaged = np.divide(hist_values[0].astype(float), [float(len(min_info_indices)), float(len(intermediate_info_indices)), float(len(max_info_indices))])
    #     hist_values_averaged_per_run[r] = hist_values_averaged
    # import pdb; pdb.set_trace()
    # hist_values_averaged_per_run_optimal_first = np.fliplr(hist_values_averaged_per_run)
    # # hist_values_norm = np.divide(hist_values_averaged_per_run_optimal_first, np.sum(hist_values_averaged_per_run_optimal_first, axis=1)[:, None])
    # mean_selected_hyps_by_lex_type = np.mean(hist_values_averaged_per_run_optimal_first, axis=0)
    # std_selected_hyps_by_lex_type = np.std(hist_values_averaged_per_run_optimal_first, axis=0)
    # conf_invs_selected_hyps_by_lex_type = stats.norm.interval(0.95, loc=mean_selected_hyps_by_lex_type, scale=std_selected_hyps_by_lex_type / np.sqrt(n_runs*n_copies))
    # conf_invs_selected_hyps_by_lex_type = np.array(conf_invs_selected_hyps_by_lex_type)
    # lower_yerr_selected_hyps_for_plot = np.subtract(mean_selected_hyps_by_lex_type, conf_invs_selected_hyps_by_lex_type[0])
    # upper_yerr_selected_hyps_for_plot = np.subtract(conf_invs_selected_hyps_by_lex_type[1], mean_selected_hyps_by_lex_type)
    # yerr_selected_hyps_for_plot = np.array([lower_yerr_selected_hyps_for_plot, upper_yerr_selected_hyps_for_plot])
    # hypothesis_count_proportions = np.divide(mean_selected_hyps_by_lex_type, np.sum(mean_selected_hyps_by_lex_type))
    # yerr_scaled_selected_hyps_for_plot = np.divide(yerr_selected_hyps_for_plot, np.sum(mean_selected_hyps_by_lex_type))

    # return hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot



def get_avg_fitness_matrix(avg_fit_matrix, n_copies, n_runs, n_iterations):
    if n_copies == 1:
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        avg_fitness_matrix_all_runs = results_dict['multi_run_avg_fitness_matrix']
    elif n_copies > 1:
        avg_fitness_matrix_all_runs = np.zeros(((n_copies*n_runs), n_iterations))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_avg_fitness_matrix = results_dict['multi_run_avg_fitness_matrix'][r]
                avg_fitness_matrix_all_runs[counter] = multi_run_avg_fitness_matrix
                counter += 1
    return avg_fitness_matrix_all_runs




def calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs):
    avg_success_per_generation = np.zeros(len(multi_run_selected_parent_indices_matrix))
    parent_perspective_index = np.where(perspective_probs==1.0)[0][0]
    parent_perspective = perspective_hyps[parent_perspective_index]
    for i in range(len(multi_run_selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = multi_run_selected_hyps_per_generation_matrix[i]
            success_per_agent = np.zeros(len(selected_hyps_per_agent))
            for a in range(len(selected_hyps_per_agent)):
                selected_hyp_index_agent = selected_hyps_per_agent[a]
                selected_hyp_agent = hypothesis_space[int(selected_hyp_index_agent)]
                learner_p_hyp = perspective_hyps[selected_hyp_agent[0]]
                if learner_p_hyp == parent_perspective:
                    success_per_agent[a] = 1.
                else:
                    success_per_agent[a] = 0.
            avg_success_per_generation[i] = np.mean(success_per_agent)
    return avg_success_per_generation



def calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level_parent, pragmatic_level_learner, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, production_error,
        learning_types, agent_type, pop_size, error, random_parent, n_meanings, n_signals,
        optimality_alpha, extra_error, n_utterances):
    avg_success_per_generation = np.zeros(len(multi_run_selected_parent_indices_matrix))
    parent_perspective_index = np.where(perspective_probs==1.0)[0][0]
    parent_perspective = perspective_hyps[parent_perspective_index]
    learning_type_index = np.where(learning_type_probs==1.0)[0][0]
    learning_type = learning_types[learning_type_index]
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
    composite_log_prior = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior)
    for i in range(len(multi_run_selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = multi_run_selected_hyps_per_generation_matrix[i]
            if random_parent:
                random_parent_index = np.random.choice(len(multi_run_selected_hyps_per_generation_matrix))
                selected_hyps_per_parent = multi_run_selected_hyps_per_generation_matrix[random_parent_index]
            else:
                selected_hyps_per_parent = multi_run_selected_hyps_per_generation_matrix[i-1]
            selected_parent_indices_per_agent = multi_run_selected_parent_indices_matrix[i]
            success_per_agent = np.zeros(len(selected_hyps_per_agent))
            for a in range(len(selected_hyps_per_agent)):
                parent_index = selected_parent_indices_per_agent[a]
                selected_hyp_index_agent = selected_hyps_per_agent[a]
                selected_hyp_agent = hypothesis_space[int(selected_hyp_index_agent)]
                learner_lex_matrix = lexicon_hyps[selected_hyp_agent[1]]
                learner_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=learner_lex_matrix)
                selected_hyp_index_parent = selected_hyps_per_parent[int(parent_index)]
                selected_hyp_parent = hypothesis_space[int(selected_hyp_index_parent)]
                parent_lex_matrix = lexicon_hyps[selected_hyp_parent[1]]
                parent_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=parent_lex_matrix)
                if pragmatic_level_parent == 'literal' or pragmatic_level_parent == 'perspective-taking':
                    parent = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, 'sample')
                elif pragmatic_level_parent == 'prag':
                    parent = pop.PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, learning_type, pragmatic_level_parent, pragmatic_level_learner, optimality_alpha, extra_error)
                if pragmatic_level_learner == 'literal' or pragmatic_level_learner == 'perspective-taking':
                    learner = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, learning_type)
                elif pragmatic_level_learner == 'prag':
                    learner = pop.PragmaticAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, learning_type, pragmatic_level_parent, pragmatic_level_learner, optimality_alpha, extra_error)
                context_matrix = context.gen_context_matrix('continuous', n_meanings, n_meanings, n_interactions)
                ca = learner.calc_comm_acc(context_matrix, communication_type, ca_measure_type, n_interactions, n_utterances, parent, learner, n_meanings, n_signals, sal_alpha, production_error, parent.pragmatic_level, speaker_p_hyp=parent.perspective, speaker_type_hyp=parent.pragmatic_level)
                success_per_agent[a] = ca
            avg_success_per_generation[i] = np.mean(success_per_agent)
    return avg_success_per_generation


def unpickle_and_calc_success(directory, filename, n_copies, success_type,
        pragmatic_level, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, production_error,
        learning_types, agent_type, pop_size, error, random_parent, n_meanings, n_signals,
        optimality_alpha, extra_error, n_utterances, n_runs, n_iterations):
    if n_copies == 1:
        pickle_filename_all_results = 'Results_'+filename+copy_specification
        results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
        multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix']
        multi_run_selected_parent_indices_matrix = results_dict['multi_run_selected_parent_indices_matrix']
        if success_type == 'communication':
            avg_success_matrix_all_runs = calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level, pragmatic_level, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, error)
        elif success_type == 'p_taking':
            avg_success_matrix_all_runs = calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs)
    elif n_copies > 1:
        avg_success_matrix_all_runs = np.zeros(((n_copies*n_runs), n_iterations))
        counter = 0
        for c in range(1, n_copies+1):
            pickle_filename_all_results = 'Results_'+filename+'_c'+str(c)
            results_dict = pickle.load(open(directory+pickle_filename_all_results+'.p', 'rb'))
            for r in range(n_runs):
                multi_run_selected_hyps_per_generation_matrix = results_dict['multi_run_selected_hyps_per_generation_matrix'][r]
                multi_run_selected_parent_indices_matrix = results_dict['multi_run_selected_parent_indices_matrix'][r]
                if success_type == 'p_taking':
                    avg_success_timecourse = calc_p_taking_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs)
                elif success_type == 'communication':
                    avg_success_timecourse = calc_communication_success(multi_run_selected_hyps_per_generation_matrix, multi_run_selected_parent_indices_matrix, pragmatic_level, pragmatic_level, communication_type, ca_measure_type, n_interactions, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspective_probs, learner_perspective, learning_type_probs, sal_alpha, error)

                avg_success_matrix_all_runs[counter] = avg_success_timecourse
                counter += 1
    return avg_success_matrix_all_runs






def calc_mean_and_conf_invs_fitness_over_gens(avg_fitness_matrix_all_runs, n_runs, n_copies):
    mean_avg_fitness_over_gens = np.mean(avg_fitness_matrix_all_runs, axis=0)
    std_avg_fitness_over_gens = np.std(avg_fitness_matrix_all_runs, axis=0)
    conf_intervals_fitness_over_gens = stats.norm.interval(0.95, loc=mean_avg_fitness_over_gens, scale=std_avg_fitness_over_gens / np.sqrt(n_runs*n_copies))
    conf_intervals_fitness_over_gens = np.array(conf_intervals_fitness_over_gens)
    lower_yerr_fitness_over_gens_for_plot = np.subtract(mean_avg_fitness_over_gens, conf_intervals_fitness_over_gens[0])
    upper_yerr_fitness_over_gens_for_plot = np.subtract(conf_intervals_fitness_over_gens[1], mean_avg_fitness_over_gens)
    yerr_fitness_over_gens_for_plot = np.array([lower_yerr_fitness_over_gens_for_plot, upper_yerr_fitness_over_gens_for_plot])
    return mean_avg_fitness_over_gens, yerr_fitness_over_gens_for_plot



def calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs):
    percentile_25_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 25, axis=0)
    median_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 50, axis=0)
    percentile_75_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 75, axis=0)
    percentiles_fitness_over_gens = np.array([percentile_25_fitness_over_gens, median_fitness_over_gens, percentile_75_fitness_over_gens])
    return percentiles_fitness_over_gens
