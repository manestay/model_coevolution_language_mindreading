"""
In the top down approach, we suppose that that each community A and community B are associated with
n signals each (with n meanings in the context). Each learner has a (n x 2n) lexicon hypothesis space,
which is initialized to fully ambiguous for their own (n x n) subset. In the isolation setting, we
expect them to learn only the n x n lexicon of their community. With interaction, we expect them to
learn the (n)x(2n) lexicon.
"""

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
from numpy.lib.function_base import _parse_input_dimensions

from hypspace import list_hypothesis_space
import lex
import measur
import plots
import pop
import prior
from lib import get_helpful_contexts, get_lexicon_hyps, get_hypothesis_space, read_config, \
                get_hyp_inds, get_lexicon_probs_ambig, calc_lex_hyp_proportions
from plot_lib import calc_p_taking_success, calc_communication_success, \
                     plot_informativeness_over_gens, plot_success_over_gens, plot_informativeness_3


parser = argparse.ArgumentParser()
parser.add_argument('config', help='path to config file')
parser.add_argument("run_name", help='name of run (name of bottom-level save dir)')
parser.add_argument('--overwrite', '-o', action='store_true')
parser.add_argument("-q", "--quiet", dest='verbose', action="store_false", help="decrease output verbosity")
parser.add_argument("-po", "--plot-only", action="store_true", help="load and plot")
parser.add_argument('--interaction_matrix', '-im', nargs='*', default=[], type=float)
parser.add_argument('--prestige', '-p', default=None, type=float)


def run_iteration(n_meanings, n_signals, n_iterations, report_every_i, turnover_type, selection_type,
        selection_weighting, communication_type, ca_measure_type, n_interactions, n_contexts, n_utterances,
        context_generation, context_type, context_size, helpful_contexts, pop_size, teacher_type,
        perspectives, sal_alpha, lexicon_probs, error, extra_error, pragmatic_level,
        optimality_alpha, learning_types, learning_type_probs, hypothesis_space, perspective_hyps,
        lexicon_hyps, perspective_prior_type, perspective_prior_strength,
        lexicon_prior_type, lexicon_prior_constant, communities, community_probs,
        interaction_matrix, lang_tups, prestige, run_idx=0):
    run_selected_hyps_per_generation_matrix, run_avg_fitness_matrix, run_parent_probs_matrix, \
            run_selected_parent_indices_matrix, run_parent_lex_indices_matrix = [], [], [], [], []
    communities_per_agent_matrix = []
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

    population = pop.BilingualPopulation(pop_size, n_meanings, n_signals, hypothesis_space, perspective_hyps,
            lexicon_hyps, perspective_prior_type, perspective_prior_strength, lexicon_prior_type,
            lexicon_prior_constant, perspectives, sal_alpha, lexicon_probs, error,
            extra_error, pragmatic_level, optimality_alpha, n_contexts, context_type, context_generation,
            context_size, helpful_contexts, n_utterances, learning_types, learning_type_probs,
            communities, community_probs, interaction_matrix, prestige)
    communities_per_agent_matrix.append(population.communities_per_agent)


    for i in range(n_iterations):
        parent_perspective = population.perspective
        random_seed = run_idx * 1000 + i
        if i == 0 or i % report_every_i == 0:
            print('iteration = {}'.format(i)),
        np.random.seed(random_seed)

        selected_hyp_per_agent_matrix, avg_fitness, parent_probs, selected_parent_indices, \
        parent_lex_indices, communities_per_agent = population.pop_update(
            context_generation, helpful_contexts, n_meanings, n_signals, error,
            selection_type, selection_weighting, communication_type, ca_measure_type, n_interactions,
            teacher_type, perspectives_per_agent=None, n_procs=n_procs)

        run_selected_hyps_per_generation_matrix.append(selected_hyp_per_agent_matrix)
        run_avg_fitness_matrix.append(avg_fitness)
        run_parent_probs_matrix.append(parent_probs)
        run_selected_parent_indices_matrix.append(selected_parent_indices)
        run_parent_lex_indices_matrix.append(parent_lex_indices)
        communities_per_agent_matrix.append(communities_per_agent)
        # community indices
        commA_inds = np.where(communities_per_agent == 'commA')
        commB_inds = np.where(communities_per_agent == 'commB')

        # calculate proportions per language
        hyp_space = population.hypothesis_space
        print()
        for comm_name, comm_inds in (('commA', commA_inds), ('commB', commB_inds)):
            fst = []
            print(comm_name)
            selected_hyps_comm = selected_hyp_per_agent_matrix[comm_inds]
            for lang_name, info_dict, null_min_max_info in lang_tups:
                hyp_inds = get_hyp_inds(selected_hyps_comm, hyp_space)
                persp_hyps = []
                for hyp in selected_hyps_comm:
                    persp_hyps.append(hyp_space[hyp][0])
                persp_prop = persp_hyps.count(parent_perspective) / float(len(persp_hyps))
                print('{} | persp: {}'.format(lang_name.ljust(5), persp_prop)),

                lex_props = calc_lex_hyp_proportions(hyp_inds, info_dict, null_min_max_info)
                print(' | lex: {}'.format(lex_props))
                fst.append(lex_props[0])

    return run_selected_hyps_per_generation_matrix, run_avg_fitness_matrix, run_parent_probs_matrix, \
        run_selected_parent_indices_matrix, run_parent_lex_indices_matrix, communities_per_agent_matrix



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
    if args.interaction_matrix:
        config_d['community.interaction_matrix'] = args.interaction_matrix
    if args.prestige:
        config_d['community.prestige'] = args.prestige
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

        communities = config.get('community', 'communities').split(' ')
        community_probs = config.get('community', 'community_probs')
        community_probs = [float(x) for x in community_probs.split(' ')]

        # cmdline priority, then config file
        if args.interaction_matrix:
            interaction_matrix = args.interaction_matrix
        else:
            interaction_matrix = config.get('community', 'interaction_matrix').split(' ')
        interaction_matrix = np.reshape([float(x) for x in interaction_matrix], (2,2))
        interaction_matrix = {comm: interaction_matrix[i] for i, comm in enumerate(communities)}
        if args.prestige:
            prestige = args.prestige
        else:
            try:
                prestige = config.getfloat('community', 'prestige')
            except (ConfigParser.NoOptionError, ValueError):
                print('no prestige specified, not using')
                prestige = None


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
        learning_types = config.get('pop_makeup', 'learning_types')
        learning_types = learning_types.split(' ')
        learning_type_probs = config.get('pop_makeup', 'learning_type_probs')
        learning_type_probs = [float(x) for x in learning_type_probs.split(' ')]
        learner_type = config.get('pop_makeup', 'learner_type')
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
        which_hyps_on_graph = config.get('simulation', 'which_hyps_on_graph')
        lex_measure = config.get('simulation', 'lex_measure')
        posterior_threshold = config.getfloat('simulation', 'posterior_threshold')
        decoupling = config.getboolean('simulation', 'decoupling')
        multithread = config.getboolean('simulation', 'multithread')

    ###
    # variables inferred from config variables
    helpful_contexts = get_helpful_contexts(n_meanings)
    lexicon_hyps = get_lexicon_hyps(which_lexicon_hyps, n_meanings, n_signals)

    # for bottom-up, we add the [[0,0],[0,0]] case
    lexicon_hyps = np.vstack((np.zeros((1,n_meanings,n_signals)), lexicon_hyps))
    # grow lexicon hyp space by num communities
    # TODO: currently only works for 2 communities
    if len(communities) == 2:
        lexicon_hyps_all = []
        for lexicon in lexicon_hyps:
            for lexicon2 in lexicon_hyps:
                lexicon_ext = np.hstack((lexicon,lexicon2))
                if lexicon_ext.sum() == 0:
                    continue
                lexicon_hyps_all.append(lexicon_ext)
        lexicon_hyps_all = np.array(lexicon_hyps_all)
    else:
        print('only size 2 communities supported')
        sys.exit(-1)


    hypothesis_space = get_hypothesis_space(agent_type, perspective_hyps, lexicon_hyps_all, pop_size)

    lh_shape = lexicon_hyps_all.shape
    lexicon_hyps_all_flat = lexicon_hyps_all.reshape(lh_shape[0], lh_shape[1] * lh_shape[2])

    commA_mat = np.hstack((np.ones((n_meanings, n_signals)), np.zeros((n_meanings,n_signals))))
    ind_commA_ambig = np.where((lexicon_hyps_all_flat == commA_mat.flatten()).all(axis=1))
    lexicon_probs_commA = np.zeros(lexicon_hyps_all.shape[0])
    lexicon_probs_commA[ind_commA_ambig] = 1

    commB_mat = np.hstack((np.zeros((n_meanings, n_signals)), np.ones((n_meanings,n_signals))))
    ind_commB_ambig = np.where((lexicon_hyps_all_flat == commB_mat.flatten()).all(axis=1))
    lexicon_probs_commB = np.zeros(lexicon_hyps_all.shape[0])
    lexicon_probs_commB[ind_commB_ambig] = 1
    lexicon_probs = {'commA': lexicon_probs_commA, 'commB': lexicon_probs_commB}
    # lexicon_probs = np.array([0. for x in range(len(lexicon_hyps_all)-1)]+[1.])
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

        if not os.path.exists(out_dir_pickle):
            os.makedirs(out_dir_pickle)
        if not os.path.exists(out_dir_plots):
            os.makedirs(out_dir_plots)
        with open(os.path.join(out_dirname, 'params.json'), 'w') as f:
            json.dump(config_d, f, indent=2)

    inds = np.arange(lexicon_hyps_all.shape[0])
    min_info = 1. / n_meanings

    informativity_per_lexiconA = lex.calc_ca_all_lexicons(lexicon_hyps_all[:, :, 0:n_meanings], error, lex_measure)
    informativity_dictA = dict(zip(inds, informativity_per_lexiconA))
    maximum_informativityA = np.amax(informativity_per_lexiconA)
    null_informativityA = np.amin(informativity_per_lexiconA)
    null_min_max_infoA = (null_informativityA, min_info, maximum_informativityA)

    informativity_per_lexiconB = lex.calc_ca_all_lexicons(lexicon_hyps_all[:, :, n_meanings:], error, lex_measure)
    informativity_dictB = dict(zip(inds, informativity_per_lexiconB))
    maximum_informativityB = np.amax(informativity_per_lexiconB)
    null_informativityB = np.amin(informativity_per_lexiconB)
    null_min_max_infoB = (null_informativityB, min_info, maximum_informativityB)

    informativity_per_lexicon_all = informativity_per_lexiconA + informativity_per_lexiconB
    informativity_dict_all = dict(zip(inds, informativity_per_lexicon_all))
    maximum_informativity_all = np.amax(informativity_per_lexicon_all)
    null_informativity_all = np.amin(informativity_per_lexicon_all)
    null_min_max_info_all = (null_informativity_all, min_info * 2, maximum_informativity_all)

    lang_tups = (
        ('langA', informativity_dictA, null_min_max_infoA),
        ('langB', informativity_dictB, null_min_max_infoB),
        ('all', informativity_dict_all, null_min_max_info_all),)
    langs = [x[0] for x in lang_tups]

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
                pop_size, teacher_type, perspectives, sal_alpha, lexicon_probs,
                error, extra_error, pragmatic_level, optimality_alpha, learning_types, learning_type_probs,
                hypothesis_space, perspective_hyps, lexicon_hyps_all, perspective_prior_type,
                perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant,
                communities, community_probs, interaction_matrix, lang_tups, prestige=prestige, run_idx=r)
            run_time_mins = (time.time()-t0)/60.
            run_times_all.append(run_time_mins)
            print('run took {:.2f}min'.format(run_time_mins))

            results.append(result_tup)

        selected_hyps_per_generation_matrix = np.array([x[0] for x in results])
        avg_fitness_matrix = np.array([x[1] for x in results])
        parent_probs_matrix = np.array([x[2] for x in results])
        selected_parent_indices_matrix = np.array([x[3] for x in results])
        parent_lex_indices_matrix = np.array([x[4] for x in results])
        communities_per_agent_matrix = np.array([x[5] for x in results])

        all_results_dict = {'selected_hyps_per_generation_matrix': selected_hyps_per_generation_matrix,
                        'avg_fitness_matrix': avg_fitness_matrix,
                        'parent_probs_matrix': parent_probs_matrix,
                        'selected_parent_indices_matrix': selected_parent_indices_matrix,
                        'parent_lex_indices_matrix': parent_lex_indices_matrix,
                        'run_time_mins': run_times_all,
                        'communities_per_agent_matrix': communities_per_agent_matrix}

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
        communities_per_agent_matrix = all_results_dict['communities_per_agent_matrix']

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



    # the lowest level arrays are of dimension (n_runs * n_iterations * 3)
    proportions = {'commA': {x: [] for x in langs}, 'commB': {x: [] for x in langs}}

    selected_hyps_commA = selected_hyps_per_generation_matrix[:,:, :pop_size/2]
    selected_hyps_commB = selected_hyps_per_generation_matrix[:,:, pop_size/2:]
    comm_tups = (('commA', selected_hyps_commA), ('commB', selected_hyps_commB))

    for comm_name, selected_hyps_comm in comm_tups:
        props_comm = proportions[comm_name]

        for run in selected_hyps_comm:
            [x.append([]) for x in props_comm.values()]
            for row in run:
                for lang_name, info_dict, null_min_max_info in lang_tups:
                    hyp_inds = get_hyp_inds(row, hypothesis_space)
                    props_row = calc_lex_hyp_proportions(hyp_inds, info_dict, null_min_max_info)
                    if lang_name not in props_comm:
                        props_comm[lang_name] = []
                    props_comm[lang_name][-1].append(props_row)

        for lang_name in langs:
            props_comm[lang_name] = np.array(props_comm[lang_name])

        out_name = 'lex_dist_over_gens_{}.png'.format(comm_name)
        print('plotting to {}'.format(os.path.join(out_dir_plots, out_name)))
        plot_informativeness_3(props_comm['langA'], props_comm['langB'], props_comm['all'], \
                langs, out_dir_plots, out_name, comm_name)

    np.random.seed(0)

    learner_perspective = 0

    comm_inds = [('commA', 0, 5), ('commB', 5, 10)]

    community_list = communities_per_agent_matrix[0][0]

    for comm_name, start_ind, end_ind in comm_inds:
        avg_pt_success_per_gen = np.zeros(n_iterations)
        avg_ca_success_per_gen = np.zeros(n_iterations)
        i=0
        for selected_hyps_per_generation, selected_parent_indices in \
                zip(selected_hyps_per_generation_matrix, selected_parent_indices_matrix):

            pt_success = calc_p_taking_success(selected_hyps_per_generation[:, start_ind:end_ind], selected_parent_indices, hypothesis_space, perspective_hyps)
            avg_pt_success_per_gen += pt_success

            # comm_success = calc_communication_success(selected_hyps_per_generation, selected_parent_indices,
            #         communication_type, ca_measure_type, n_interactions,
            #         hypothesis_space, perspective_hyps, lexicon_hyps_all, perspective_prior_type,
            #         perspective_prior_strength, lexicon_prior_type, lexicon_prior_constant,
            #         learner_perspective, learning_types, learning_type_probs, sal_alpha, error,
            #         agent_type, pop_size, n_meanings, n_signals, n_utterances, start_ind, end_ind,
            #         communities, community_list, prestige)
            # avg_ca_success_per_gen += comm_success

            learner_perspective = 1 - learner_perspective
            i +=1

        # print('perspective-inference performance over generations:')
        avg_pt_success_per_gen /= n_runs
        # print('communication performance over generations:')
        # avg_ca_success_per_gen /= n_runs

        out_name = 'success_over_gens_{}.png'.format(comm_name)
        print('plotting to {}'.format(os.path.join(out_dir_plots, out_name)))
        plot_success_over_gens(avg_pt_success_per_gen, avg_ca_success_per_gen, out_dir_plots,  out_name)
