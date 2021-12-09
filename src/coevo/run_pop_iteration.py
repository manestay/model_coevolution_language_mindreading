import argparse
from collections import OrderedDict
import ConfigParser
import logging
import time
import cPickle as pickle
import os
import json
import sys

import numpy as np

from hypspace import list_hypothesis_space
import lex
import measur
import plots
import pop
import prior
from lib import get_helpful_contexts, get_lexicon_hyps, get_hypothesis_space, read_config, \
                get_hyp_inds
from plot_lib import calc_lex_hyp_proportions, calc_p_taking_success, calc_communication_success, \
                     plot_informativeness_over_gens, plot_success_over_gens


parser = argparse.ArgumentParser()
parser.add_argument('config', help='path to config file')
parser.add_argument("run_name", help='name of run (name of bottom-level save dir)')
parser.add_argument('--overwrite', '-o', action='store_true')
parser.add_argument("-q", "--quiet", dest='verbose', action="store_false", help="decrease output verbosity")
parser.add_argument("-po", "--plot-only", action="store_true", help="load and plot (make sure you ran this config before)")


def run_iteration(n_meanings, n_signals, n_iterations, report_every_i, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts, pop_size, teacher_type, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, recording, print_pop=False, run_idx=0):
    run_selected_hyps_per_generation_matrix, run_avg_fitness_matrix, run_parent_probs_matrix, run_selected_parent_indices_matrix, run_parent_lex_indices_matrix = [], [], [], [], []
    if multithread and turnover_type == 'chain':
        print('cannot multithread with `turnover_type` == "chain", only using 1 process')
        n_procs = 1
    elif multithread:
        n_procs = pop_size
        print('`multithread` option is true -- {} agents in parallel updated per iteration'.format(
            n_procs
        ))
    else:
        n_procs = 1

    # 1) First the initial population is created:

    if agent_type == 'p_distinction':
        # 1.1) Then the prior probability distribution for all agents is created:
        ## 1.1.1) First the perspective prior is created:
        perspective_prior = prior.create_perspective_prior(perspective_hyps, lexicon_hyps, perspective_prior_type, learner_perspective, perspective_prior_strength)
        ## 1.1.2) Then the lexicon prior is created:
        lexicon_prior = prior.create_lexicon_prior(lexicon_hyps, lexicon_prior_type, lexicon_prior_constant, error)
        ## 1.1.3) And finally the full composite prior matrix is created using the separate lexicon_prior and perspective_prior, and following the configuration of hypothesis_space
        composite_log_prior = prior.list_composite_log_priors_with_speaker_distinction(hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior, lexicon_prior, pop_size)

        # 1.2) Then the population is created:

        lexicons_per_agent = np.random.choice(np.arange(len(lexicon_hyps)), pop_size, replace=True, p=lexicon_probs)

        perspectives_per_agent = np.random.choice(perspectives, pop_size, replace=True, p=perspective_probs)

        for i in range(pop_size):
            learning_types_per_agent = np.random.choice(learning_types, pop_size, replace=True, p=learning_type_probs)

        population = pop.DistinctionPopulation(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, composite_log_prior, perspectives, perspectives_per_agent, perspective_probs, sal_alpha, lexicon_hyps, lexicons_per_agent, error, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_types_per_agent, learning_type_probs)

    elif agent_type == 'no_p_distinction':
        perspectives_per_agent = None
        population = pop.Population(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, perspectives, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation, context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs)
    if print_pop:
        print('initial population for run 0 is: {}')
        population.print_population()

    for i in range(n_iterations):
        parent_perspective = population.perspective
        random_seed = run_idx * 1000 + i
        if i == 0 or i % report_every_i == 0:
            print('iteration = {}'.format(i)),
        selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, parent_lex_indices, log_posteriors = population.pop_update(recording, context_generation, helpful_contexts, n_meanings, n_signals, error, turnover_type, selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions, teacher_type, perspectives_per_agent=perspectives_per_agent, n_procs=n_procs, seed=random_seed)

        run_selected_hyps_per_generation_matrix.append(selected_hyp_per_agent_matrix)
        run_avg_fitness_matrix.append(avg_fitness)
        run_parent_probs_matrix.append(parent_probs)
        run_selected_parent_indices_matrix.append(selected_parent_indices)
        run_parent_lex_indices_matrix.append(parent_lex_indices)

        hyp_space = population.hypothesis_space
        hyp_inds = get_hyp_inds(selected_hyp_per_agent_matrix, argsort_informativity_per_lexicon, hyp_space)

        persp_hyps = []
        for hyp in selected_hyp_per_agent_matrix:
            persp_hyps.append(hyp_space[hyp][0])
        persp_prop = persp_hyps.count(parent_perspective) / float(len(persp_hyps))
        print(' | persp prop is {}'.format(persp_prop)),
        props = calc_lex_hyp_proportions(hyp_inds, min_info_indices, intermediate_info_indices, max_info_indices)
        print(' | lex props are {}'.format(props))

    return run_selected_hyps_per_generation_matrix, run_avg_fitness_matrix, run_parent_probs_matrix, run_selected_parent_indices_matrix, run_parent_lex_indices_matrix



if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    config = ConfigParser.ConfigParser()
    print('loading config file...')
    config.read(args.config)
    config_d = OrderedDict()
    if args.verbose:
        logging.debug('config parameters:')
        for section in config.sections():
            print('CONFIG {}'.format(section))
            for k,v in config.items(section):
                print('{} = {} ,'.format(k,v)),
                config_d['{}.{}'.format(section, k)] = v
            print('')
            print('-' * 20)
    # read_config(args.config)

    ###
    # load config values
    print('loading config variables...')
    if True: # for IDE collapsing this section :)
        root_path = config.get('paths', 'root_path')
        run_type_dir = config.get('paths', 'run_type_dir')

        n_meanings = config.getint('lexicon', 'n_meanings')
        n_signals = config.getint('lexicon', 'n_signals')
        n_utterances = config.getint('lexicon', 'n_utterances')
        n_contexts = config.getint('lexicon', 'n_contexts')
        n_iterations = config.getint('lexicon', 'n_iterations')
        n_runs = config.getint('lexicon', 'n_runs')

        context_generation = config.get('context', 'context_generation')
        context_type = config.get('context', 'context_type')
        context_size = config.getint('context', 'context_size')
        sal_alpha = config.getfloat('context', 'sal_alpha')
        error = config.getfloat('context', 'error')
        extra_error = config.getboolean('context', 'extra_error')

        pop_size = config.getint('population', 'pop_size')
        agent_type = config.get('population', 'agent_type')
        pragmatic_level = config.get('population', 'pragmatic_level')
        optimality_alpha = config.getfloat('population', 'optimality_alpha')
        teacher_type = config.get('population', 'teacher_type')

        perspective_hyps = config.get('hypothesis', 'perspective_hyps')
        perspective_hyps = [int(x) for x in perspective_hyps.split(' ')]
        which_lexicon_hyps = config.get('hypothesis', 'which_lexicon_hyps')

        perspectives = config.get('pop_makeup', 'perspectives')
        perspectives = [int(x) for x in perspectives.split(' ')]
        perspective_probs = config.get('pop_makeup', 'perspective_probs')
        perspective_probs = [float(x) for x in perspective_probs.split(' ')]
        learning_types = config.get('pop_makeup', 'learning_types')
        learning_types = learning_types.split(' ')
        learning_type_probs = config.get('pop_makeup', 'learning_type_probs')
        learning_type_probs = [float(x) for x in learning_type_probs.split(' ')]
        learner_type = config.get('pop_makeup', 'learner_type')
        learner_perspective = config.getint('pop_makeup', 'learner_perspective')
        perspective_prior_type = config.get('pop_makeup', 'perspective_prior_type')
        perspective_prior_strength = config.getfloat('pop_makeup', 'perspective_prior_strength')
        lexicon_prior_type = config.get('pop_makeup', 'lexicon_prior_type')
        lexicon_prior_constant = config.getfloat('pop_makeup', 'lexicon_prior_constant')

        speaker_order_type = config.get('learner', 'speaker_order_type')
        first_input_stage_ratio = config.getfloat('learner', 'first_input_stage_ratio')

        run_type = config.get('simulation', 'run_type')
        communication_type = config.get('simulation', 'communication_type')
        ca_measure_type = config.get('simulation', 'ca_measure_type')
        n_interactions = config.getint('simulation', 'n_interactions')
        selection_type = config.get('simulation', 'selection_type')
        try:
            selection_weighting = config.getfloat('simulation', 'selection_weighting')
        except ValueError:
            selection_weighting = 'none'
        turnover_type = config.get('simulation', 'turnover_type')
        report_every_i = config.getint('simulation', 'report_every_i')
        cut_off_point = config.getint('simulation', 'cut_off_point')
        report_every_r = config.getint('simulation', 'report_every_r')
        recording = config.get('simulation', 'recording')
        which_hyps_on_graph = config.get('simulation', 'which_hyps_on_graph')
        lex_measure = config.get('simulation', 'lex_measure')
        posterior_threshold = config.getfloat('simulation', 'posterior_threshold')
        decoupling = config.getboolean('simulation', 'decoupling')
        multithread = config.getboolean('simulation', 'multithread')

    ###
    # variables inferred from config variables
    helpful_contexts = get_helpful_contexts(n_meanings)
    lexicon_hyps = get_lexicon_hyps(which_lexicon_hyps, n_meanings, n_signals)
    hypothesis_space = get_hypothesis_space(agent_type, perspective_hyps, lexicon_hyps, pop_size)
    lexicon_probs = np.array([0. for x in range(len(lexicon_hyps)-1)]+[1.])

    ###
    out_dirname = os.path.join(root_path, run_type_dir, args.run_name)
    out_dir_pickle = os.path.join(out_dirname, 'pickles')
    out_dir_plots = os.path.join(out_dirname, 'plots')

    if not args.plot_only:
        if os.path.exists(out_dirname):
            if not args.overwrite:
                print('{} exists, pass --overwrite to continue. exiting'.format(out_dirname))
                sys.exit(-1)
            else:
                print('overwriting existing run...')
        else:
            os.makedirs(out_dirname)
            os.makedirs(out_dir_pickle)
            os.makedirs(out_dir_plots)
        with open(os.path.join(out_dirname, 'params.json'), 'w') as f:
            json.dump(config_d, f, indent=2)



    ###
    # CATEGORISING LEXICONS BY INFORMATIVENESS BELOW
    logging.debug("lexicon_hyps.shape: {}".format(lexicon_hyps.shape))
    informativity_per_lexicon = lex.calc_ca_all_lexicons(lexicon_hyps, error, lex_measure)
    logging.debug("informativity_per_lexicon.shape: {}".format(informativity_per_lexicon.shape))

    argsort_informativity_per_lexicon = np.argsort(informativity_per_lexicon)
    logging.debug("argsort_informativity_per_lexicon.shape: {}".format(argsort_informativity_per_lexicon.shape))

    informativity_per_lexicon_sorted = np.round(informativity_per_lexicon[argsort_informativity_per_lexicon], decimals=2)
    logging.debug("informativity_per_lexicon_sorted.shape: {}".format(informativity_per_lexicon_sorted.shape))

    unique_informativity_per_lexicon = np.unique(informativity_per_lexicon_sorted)
    logging.debug("unique_informativity_per_lexicon.shape: {}".format(unique_informativity_per_lexicon.shape))

    minimum_informativity = np.amin(informativity_per_lexicon_sorted)
    logging.debug("minimum_informativity: {}".format(minimum_informativity))

    min_info_indices = np.argwhere(informativity_per_lexicon_sorted==minimum_informativity)
    logging.debug("min_info_indices: {}".format(min_info_indices.flatten()))

    maximum_informativity = np.amax(informativity_per_lexicon_sorted)
    logging.debug("maximum_informativity: {}".format(maximum_informativity))

    max_info_indices = np.argwhere(informativity_per_lexicon_sorted==maximum_informativity)
    logging.debug("max_info_indices: {}".format(max_info_indices.flatten()))

    intermediate_info_indices = np.arange(min_info_indices[-1]+1, max_info_indices[0])

    lexicon_hyps_sorted = lexicon_hyps[argsort_informativity_per_lexicon]
    logging.debug("lexicon_hyps_sorted.shape: {}".format(lexicon_hyps_sorted.shape))

    pickle_file_title_all_results = os.path.join(out_dir_pickle, 'results.p')

    if not args.plot_only:
        all_results = {}
        results = []
        run_times_all = []
        start_all_runs = time.time()
        for r in range(n_runs):
            if r == 0 or r % report_every_r == 0:
                print('run = {}'.format(r))

            t0 = time.time()
            result_tup = run_iteration(n_meanings, n_signals, n_iterations, report_every_i, turnover_type,
                selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions,
                n_contexts, n_utterances, context_generation, context_type, context_size, helpful_contexts,
                pop_size, teacher_type, agent_type, perspectives, perspective_probs, sal_alpha, lexicon_probs,
                error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs,
                hypothesis_space, perspective_hyps, lexicon_hyps, learner_perspective, perspective_prior_type,
                perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant, recording,
                print_pop=(r==0), run_idx=r)
            run_time_mins = (time.time()-t0)/60.
            run_times_all.append(run_time_mins)
            print('run took {:.2f}min'.format(run_time_mins))

            results.append(result_tup)

        selected_hyps_per_generation_matrix = np.array([x[0] for x in results])
        avg_fitness_matrix = np.array([x[1] for x in results])
        parent_probs_matrix = np.array([x[2] for x in results])
        selected_parent_indices_matrix = np.array([x[3] for x in results])
        parent_lex_indices_matrix = np.array([x[4] for x in results])

        all_results_dict = {'selected_hyps_per_generation_matrix': selected_hyps_per_generation_matrix,
                        'avg_fitness_matrix': avg_fitness_matrix,
                        'parent_probs_matrix': parent_probs_matrix,
                        'selected_parent_indices_matrix': selected_parent_indices_matrix,
                        'parent_lex_indices_matrix': parent_lex_indices_matrix,
                        'run_time_mins':run_times_all}

        run_time_all_mins = (time.time()-start_all_runs)/60.
        print('all runs took {:.2f}min total'.format(run_time_all_mins))

    else:
        print("loading, not new run")
        all_results_dict = pickle.load(open(pickle_file_title_all_results, 'rb'))
        print('loaded')
        selected_hyps_per_generation_matrix = all_results_dict['selected_hyps_per_generation_matrix']
        avg_fitness_matrix = all_results_dict['avg_fitness_matrix']
        parent_probs_matrix = all_results_dict['parent_probs_matrix']
        selected_parent_indices_matrix = all_results_dict['selected_parent_indices_matrix']
        parent_lex_indices_matrix = all_results_dict['parent_lex_indices_matrix']
        run_times_all = all_results_dict['run_time_mins']

    #####
    # post processing stuff

    proportion_max_offspring_single_parent_matrix = np.zeros((n_runs, n_iterations))
    for r in range(n_runs):
        for i in range(n_iterations):
            parent_indices = selected_parent_indices_matrix[r][i]
            parent_index_counts = np.bincount(parent_indices.astype(int))
            max_offspring_single_parent = np.amax(parent_index_counts)
            proportion_max_offspring_single_parent = np.divide(max_offspring_single_parent.astype(float), float(pop_size))
            proportion_max_offspring_single_parent_matrix[r][i] = proportion_max_offspring_single_parent

    ######
    # save to pickles
    if not args.plot_only:
        pickle.dump(all_results_dict, open(pickle_file_title_all_results, 'wb'))

        pickle_file_title_max_offspring_single_parent = os.path.join(out_dir_pickle, 'max_offspring.p')

        pickle.dump(proportion_max_offspring_single_parent_matrix, open(pickle_file_title_max_offspring_single_parent, 'wb'))


    ######
    # statistics for plotting

    # dimensions are (run_idx x iteration x agent)
    selected_hyps_new_lex_order_all_runs = []
    proportions = []
    for run in selected_hyps_per_generation_matrix:
        selected_hyps_new_lex_order_all_runs.append([])
        proportions.append([])
        for row in run:
            hyp_inds = get_hyp_inds(row, argsort_informativity_per_lexicon, hypothesis_space)
            selected_hyps_new_lex_order_all_runs[-1].append(hyp_inds)
            props = calc_lex_hyp_proportions(hyp_inds, min_info_indices, intermediate_info_indices, max_info_indices)
            proportions[-1].append(props)
    selected_hyps_new_lex_order_all_runs = np.array(selected_hyps_new_lex_order_all_runs)
    proportions = np.array(proportions)
    # for i, props in enumerate(proportions.mean(axis=0)):
    #     print('run {} proportions are {}'.format(i, props))

    # hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot = calc_mean_and_conf_invs_distribution(n_runs, 1, lexicon_hyps, which_hyps_on_graph, min_info_indices, intermediate_info_indices, max_info_indices, n_iterations, cut_off_point, selected_hyps_new_lex_order_all_runs)

    # print("hypothesis_count_proportions are: {}".format(hypothesis_count_proportions))
    # print("yerr_scaled_selected_hyps_for_plot is: {}".format(yerr_scaled_selected_hyps_for_plot))

    ###
    # do plotting
    # plot_title = 'Egocentric perspective prior & '+str(n_meanings)+'x'+str(n_signals)+' lexicons'
    # plots.plot_lex_distribution(out_dir_plots, 'inform_.png', plot_title, hypothesis_count_proportions, yerr_scaled_selected_hyps_for_plot, cut_off_point, text_size=1.6)
    print('plotting to {}'.format(os.path.join(out_dir_plots, 'lex_dist_over_gens.png')))
    plot_informativeness_over_gens(proportions, out_dir_plots, 'lex_dist_over_gens.png')

    avg_pt_success_per_gen = np.zeros(n_iterations)
    avg_ca_success_per_gen = np.zeros(n_iterations)
    np.random.seed(0)
    for selected_hyps_per_generation, selected_parent_indices in \
            zip(selected_hyps_per_generation_matrix, selected_parent_indices_matrix):
        pt_success = calc_p_taking_success(selected_hyps_per_generation, selected_parent_indices, hypothesis_space, perspective_hyps)
        avg_pt_success_per_gen += pt_success

        comm_success = calc_communication_success(selected_hyps_per_generation, selected_parent_indices,
                pragmatic_level, pragmatic_level, communication_type, ca_measure_type, n_interactions,
                hypothesis_space, perspective_hyps, lexicon_hyps, perspective_prior_type,
                perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant,
                learner_perspective, learning_types, learning_type_probs, sal_alpha, error, agent_type,
                pop_size, n_meanings, n_signals, n_utterances)
        avg_ca_success_per_gen += comm_success
    # print('perspective-inference performance over generations:')
    avg_pt_success_per_gen /= n_runs
    # print('communication performance over generations:')
    avg_ca_success_per_gen /= n_runs

    print('plotting to {}'.format(os.path.join(out_dir_plots, 'success_over_gens.png')))
    plot_success_over_gens(avg_pt_success_per_gen, avg_ca_success_per_gen, out_dir_plots,  'success_over_gens.png')
