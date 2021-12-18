import cPickle as pickle

import logging

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import context
import lex
import prior
import pop
import measur
from lib import normalize

logging.getLogger('matplotlib').setLevel(logging.WARNING)
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

def calc_percentiles_fitness_over_gens(avg_fitness_matrix_all_runs):
    percentile_25_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 25, axis=0)
    median_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 50, axis=0)
    percentile_75_fitness_over_gens = np.percentile(avg_fitness_matrix_all_runs, 75, axis=0)
    percentiles_fitness_over_gens = np.array([percentile_25_fitness_over_gens, median_fitness_over_gens, percentile_75_fitness_over_gens])
    return percentiles_fitness_over_gens

def calc_lex_hyp_proportions(hyp_inds, info_dict, min_max_info):
    min_info, max_info = min_max_info
    props = [0] * 3
    for hyp_ind in hyp_inds:
        info = info_dict[hyp_ind]
        if info == max_info:
            props[0] += 1
        elif info == min_info:
            props[2] += 1
        else:
            props[1] += 1
    return normalize(np.array(props))

def plot_informativeness_over_gens(proportions, plot_file_path, plot_file_title):
    optimal = proportions[:, :, 0].mean(axis=0)
    partial = proportions[:, :, 1].mean(axis=0)
    uninform = proportions[:, :, 2].mean(axis=0)
    gens = np.arange(0, proportions.shape[1])

    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=1.6)
    lex_type_labels = ['Optimal', 'Partly informative', 'Uninformative']
    ind = np.arange(len(lex_type_labels))
    width = 0.25
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
    ax.plot(gens, optimal, 'b', linewidth=3, label='optimal')
    ax.plot(gens, partial, 'c', linewidth=3, label='partial')
    ax.plot(gens, uninform, 'g', linewidth=3, label='uninformative')

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Mean Proportion')
    fname = os.path.join(plot_file_path, plot_file_title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def plot_informativeness_3(props_A, propsB, props_all, langs, plot_file_path, plot_file_title, plot_title=None):
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(3, figsize=(8,10))
    if plot_title:
        fig.suptitle(plot_title, fontsize=20)
    # fig.subplots_adjust(top=2)
    for i, (proportions, name) in enumerate(zip((props_A, propsB, props_all), langs)):
        ax = axs[i]
        optimal = proportions[:, :, 0].mean(axis=0)
        partial = proportions[:, :, 1].mean(axis=0)
        uninform = proportions[:, :, 2].mean(axis=0)
        gens = np.arange(0, proportions.shape[1])

        sns.set_style("whitegrid")
        sns.set_palette("deep")
        sns.set(font_scale=1.6)
        lex_type_labels = ['Optimal', 'Partly informative', 'Uninformative']
        ind = np.arange(len(lex_type_labels))
        width = 0.25
        ax.plot(gens, optimal, 'b', linewidth=2, label='optimal')
        ax.plot(gens, partial, 'c', linewidth=2, label='partial')
        ax.plot(gens, uninform, 'g', linewidth=2, label='uninformative')

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Generations')
        ax.set_ylabel('Mean Proportion')
        ax.set_title(name,fontsize=15)
        fname = os.path.join(plot_file_path, plot_file_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(fname)
    plt.show()

def calc_p_taking_success(selected_hyps_per_generation_matrix, selected_parent_indices_matrix, hypothesis_space, perspective_hyp):
    avg_success_per_generation = np.zeros(len(selected_parent_indices_matrix))
    parent_perspective = 1
    for i in range(len(selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = selected_hyps_per_generation_matrix[i]
            success_per_agent = np.array([hypothesis_space[x][0] == parent_perspective for x in selected_hyps_per_agent])
            avg_success_per_generation[i] = np.mean(success_per_agent)
            parent_perspective = 1 - parent_perspective
    return avg_success_per_generation


def calc_communication_success(selected_hyps_per_generation_matrix, selected_parent_indices_matrix,
        communication_type, ca_measure_type, n_interactions,
        hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength,
        lexicon_prior_type, lexicon_prior_constant, learner_perspective, learning_types, learning_type_probs,
        sal_alpha, error, agent_type, pop_size, n_meanings, n_signals, n_utterances, \
        start_ind=None, end_ind=None,
        communities=None, community_list=None, prestige=None):
    if start_ind is None:
        start_ind, end_ind = 0, selected_hyps_per_generation_matrix.shape[1]
    avg_success_per_generation = np.zeros(len(selected_parent_indices_matrix))
    parent_perspective = 1
    learning_type_index = learning_type_probs.index(1.0)
    learning_type = learning_types[learning_type_index]
    perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
    lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
    composite_log_prior = prior.list_composite_log_priors(agent_type, pop_size, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior)
    for i in range(len(selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = selected_hyps_per_generation_matrix[i]
            selected_hyps_per_parent = selected_hyps_per_generation_matrix[i-1]
            selected_parent_indices_per_agent = selected_parent_indices_matrix[i]
            success_per_agent = np.zeros(len(selected_hyps_per_agent))
            for a in range(start_ind, end_ind):
                community = community_list[a] if community_list is not None else None
                parent_index = selected_parent_indices_per_agent[a]
                selected_hyp_index_agent = selected_hyps_per_agent[a]
                selected_hyp_agent = hypothesis_space[int(selected_hyp_index_agent)]
                learner_lex_matrix = lexicon_hyps[selected_hyp_agent[1]]
                learner_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=learner_lex_matrix)
                selected_hyp_index_parent = selected_hyps_per_parent[int(parent_index)]
                selected_hyp_parent = hypothesis_space[int(selected_hyp_index_parent)]
                parent_lex_matrix = lexicon_hyps[selected_hyp_parent[1]]
                parent_lexicon = lex.Lexicon('specified_lexicon', n_meanings, n_signals, ambiguous_lex=None, specified_lexicon=parent_lex_matrix)

                if community:
                    parent = pop.BilingualAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, 'sample', community, communities, prestige)
                    learner = pop.BilingualAgent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, 'sample', community, communities, prestige)
                else:
                    parent = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, parent_perspective, sal_alpha, parent_lexicon, 'sample')
                    learner = pop.Agent(perspective_hyps, lexicon_hyps, composite_log_prior, composite_log_prior, learner_perspective, sal_alpha, learner_lexicon, learning_type)
                context_matrix = context.gen_context_matrix('continuous', n_meanings, n_meanings, n_interactions)
                ca = measur.calc_comm_acc(context_matrix, communication_type, ca_measure_type, n_interactions, n_utterances, parent, learner, n_meanings, n_signals, sal_alpha, error, parent.pragmatic_level, s_p_hyp=parent.perspective, s_type_hyp=parent.pragmatic_level)
                success_per_agent[a] = ca
            avg_success_per_generation[i] = np.mean(success_per_agent)
            parent_perspective = 1 - parent_perspective
    return avg_success_per_generation

def plot_success_over_gens(avg_pt_success_per_gen, avg_ca_success_per_gen, plot_file_path, plot_file_title):
    gens = np.arange(0, len(avg_pt_success_per_gen))

    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set(font_scale=1.6)
    lex_type_labels = ['communication', 'p-inference']
    ind = np.arange(len(lex_type_labels))
    width = 0.25
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.plot(gens, avg_pt_success_per_gen, 'r', label='p-inference')
        ax.plot(gens, avg_ca_success_per_gen, 'y', label='comm_acc')

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Mean success')
    fname = os.path.join(plot_file_path, plot_file_title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
