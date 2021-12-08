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

def calc_lex_hyp_proportions(hyp_inds, min_info_indices, intermediate_info_indices, max_info_indices):
    hist_values, _ = np.histogram(hyp_inds, bins=[0, min_info_indices[-1]+1, intermediate_info_indices[-1]+1, max_info_indices[-1]])
    props = np.round(np.true_divide(hist_values,sum(hist_values)), 3)
    props = np.flip(props)
    return props

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
        ax.plot(gens, optimal, 'b', label='optimal')
        ax.plot(gens, partial, 'c', label='partial')
        ax.plot(gens, uninform, 'g', label='uninformative')

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Mean Proportion')
    fname = os.path.join(plot_file_path, plot_file_title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

def calc_p_taking_success(selected_hyps_per_generation_matrix, selected_parent_indices_matrix, hypothesis_space, perspective_hyps, perspective_probs):
    avg_success_per_generation = np.zeros(len(selected_parent_indices_matrix))
    parent_perspective_index = perspective_probs.index(1.0)
    parent_perspective = perspective_hyps[parent_perspective_index]
    for i in range(len(selected_hyps_per_generation_matrix)):
        if i == 0:
            avg_success_per_generation[i] = 'NaN'
        else:
            selected_hyps_per_agent = selected_hyps_per_generation_matrix[i]
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
