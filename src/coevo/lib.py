import numpy as np
from scipy import stats

import hypspace
import saveresults


def get_helpful_contexts(n_meanings):
    if n_meanings == 2:
        helpful_contexts = np.array([[0.1, 0.7], [0.3, 0.9],
                                    [0.7, 0.1], [0.9, 0.3]])
    elif n_meanings == 3:
        helpful_contexts = np.array([[0.1, 0.2, 0.9], [0.1, 0.8, 0.9],
                                    [0.1, 0.9, 0.2], [0.1, 0.9, 0.8],
                                    [0.2, 0.1, 0.9], [0.8, 0.1, 0.9],
                                    [0.2, 0.9, 0.1], [0.8, 0.9, 0.1],
                                    [0.9, 0.1, 0.2], [0.9, 0.1, 0.8],
                                    [0.9, 0.2, 0.1], [0.9, 0.8, 0.1]])
    elif n_meanings == 4:
        helpful_contexts = np.array([[0.1, 0.2, 0.3, 0.9], [0.1, 0.3, 0.6, 0.7],
                                    [0.1, 0.2, 0.9, 0.3], [0.1, 0.3, 0.7, 0.6],
                                    [0.1, 0.3, 0.2, 0.9], [0.1, 0.6, 0.3, 0.7],
                                    [0.1, 0.3, 0.9, 0.2], [0.1, 0.6, 0.7, 0.3],
                                    [0.1, 0.9, 0.2, 0.3], [0.1, 0.7, 0.3, 0.6],
                                    [0.1, 0.9, 0.3, 0.2], [0.1, 0.7, 0.6, 0.3],
                                    [0.2, 0.1, 0.3, 0.9], [0.3, 0.1, 0.6, 0.7],
                                    [0.2, 0.1, 0.9, 0.3], [0.3, 0.1, 0.7, 0.6],
                                    [0.2, 0.3, 0.1, 0.9], [0.3, 0.6, 0.1, 0.7],
                                    [0.2, 0.3, 0.9, 0.1], [0.3, 0.6, 0.7, 0.1],
                                    [0.2, 0.9, 0.1, 0.3], [0.3, 0.7, 0.1, 0.6],
                                    [0.2, 0.9, 0.3, 0.1], [0.3, 0.7, 0.6, 0.1],
                                    [0.3, 0.1, 0.2, 0.9], [0.6, 0.1, 0.3, 0.7],
                                    [0.3, 0.1, 0.9, 0.2], [0.6, 0.1, 0.7, 0.3],
                                    [0.3, 0.2, 0.1, 0.9], [0.6, 0.3, 0.1, 0.7],
                                    [0.3, 0.2, 0.9, 0.1], [0.6, 0.3, 0.7, 0.1],
                                    [0.9, 0.1, 0.2, 0.3], [0.6, 0.7, 0.1, 0.3],
                                    [0.9, 0.1, 0.3, 0.2], [0.6, 0.7, 0.3, 0.1],
                                    [0.9, 0.2, 0.1, 0.3], [0.7, 0.1, 0.3, 0.6],
                                    [0.9, 0.2, 0.3, 0.1], [0.7, 0.1, 0.6, 0.3],
                                    [0.9, 0.3, 0.1, 0.2], [0.7, 0.3, 0.1, 0.6],
                                    [0.9, 0.3, 0.2, 0.1], [0.7, 0.3, 0.6, 0.1]])
    return helpful_contexts


def get_lexicon_hyps(which_lexicon_hyps, n_meanings, n_signals):
    if which_lexicon_hyps == 'all':
        # The lexicon hypotheses that the learner will consider (1D numpy array)
        lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals)
    elif which_lexicon_hyps == 'all_with_full_s_space':
        all_lexicon_hyps = hypspace.create_all_lexicons(n_meanings, n_signals)
        # The lexicon hypotheses that the learner will consider (1D numpy array)
        lexicon_hyps = hypspace.remove_subset_of_signals_lexicons(all_lexicon_hyps)
    elif which_lexicon_hyps == 'only_optimal':
        # The lexicon hypotheses that the learner will consider (1D numpy array)
        lexicon_hyps = hypspace.create_all_optimal_lexicons(n_meanings, n_signals)
    return lexicon_hyps


def get_agent_type(run_type):

    if run_type == 'population_diff_pop' or run_type == 'population_same_pop':
        # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
        agent_type = 'no_p_distinction'
    elif run_type == 'population_same_pop_dist_learner':
        # This can be set to either 'p_distinction' or 'no_p_distinction'. Determines whether the population is made up of DistinctionAgent objects or Agent objects (DistinctionAgents can learn different perspectives for different agents, Agents can only learn one perspective for all agents they learn from).
        agent_type = 'p_distinction'
    return agent_type


def get_hypothesis_space(agent_type, perspective_hyps, lexicon_hyps, pop_size):
    if agent_type == 'no_p_distinction' or agent_type == 'bilingual':
        # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
        hypothesis_space = hypspace.list_hypothesis_space(perspective_hyps, lexicon_hyps)

    elif agent_type == 'p_distinction':
        # The full space of composite hypotheses that the learner will consider (2D numpy matrix with composite hypotheses on the rows, perspective hypotheses on column 0 and lexicon hypotheses on column 1)
        hypothesis_space = hypspace.list_hypothesis_space_with_speaker_distinction(
            perspective_hyps, lexicon_hyps, pop_size)
    return hypothesis_space


def get_dirname(context_generation, selection_type, n_runs, n_meanings, n_signals, n_iterations, n_contexts, n_utterances, pop_size, pragmatic_level, optimality_alpha, perspective_prior_type, lexicon_prior_type, which_lexicon_hyps, teacher_type, communication_type, ca_measure_type, helpful_contexts,
                 error, perspective_probs, perspective_prior_strength, lexicon_prior_constant, learning_types, learning_type_probs, selection_weighting):
    perspective_probs_string = saveresults.convert_array_to_string(perspective_probs)
    perspective_prior_strength_string = saveresults.convert_array_to_string(
        perspective_prior_strength)
    lexicon_prior_constant_string = saveresults.convert_array_to_string(lexicon_prior_constant)
    if learning_type_probs[0] == 1.:
        learning_type_string = learning_types[0]
    elif learning_type_probs[1] == 1.:
        learning_type_string = learning_types[1]

    if context_generation == 'random':
        if selection_type == 'none' or selection_type == 'l_learning':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+str(error)+'_'+pragmatic_level+'_a_'+str(
                optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+str(selection_weighting)+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+str(error)+'_'+pragmatic_level + \
                '_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string + \
                '_'+which_lexicon_hyps+'_l_prior_' + \
                str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string + \
                '_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(n_utterances)+'_U_'+'err_'+str(error)+'_'+pragmatic_level + \
                '_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string + \
                '_'+which_lexicon_hyps+'_l_prior_' + \
                str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string + \
                '_'+learning_type_string+'_'+teacher_type

    elif context_generation == 'only_helpful' or context_generation == 'optimal':
        if selection_type == 'none' or selection_type == 'l_learning':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+str(error)+'_'+pragmatic_level + \
                '_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string + \
                '_'+which_lexicon_hyps+'_l_prior_' + \
                str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string + \
                '_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'p_taking':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_weight_'+str(selection_weighting)+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+str(
                error)+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
        elif selection_type == 'ca_with_parent':
            dirname = 'iter_'+str(n_meanings)+'M_'+str(n_signals)+'S'+'_size_'+str(pop_size)+'_select_'+selection_type+'_'+communication_type+'_'+ca_measure_type+'_'+str(n_runs)+'_R_'+str(n_iterations)+'_I_'+str(n_contexts)+'_C_'+str(context_generation)+'_'+str(len(helpful_contexts))+'_'+str(n_utterances)+'_U_'+'err_'+str(
                error)+'_'+pragmatic_level+'_a_'+str(optimality_alpha)[0]+'_p_probs_'+perspective_probs_string+'_p_prior_'+str(perspective_prior_type)[0:4]+'_'+perspective_prior_strength_string+'_'+which_lexicon_hyps+'_l_prior_'+str(lexicon_prior_type)[0:4]+'_'+lexicon_prior_constant_string+'_'+learning_type_string+'_'+teacher_type
    return dirname

def read_config(fname):
    # helper function to print python code to load config to variables
    # some post-processing required for non-str types and lists!
    section = ''
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            elif line.startswith('['):
                section = line.strip('\n[]')
                continue
            key, value = [x.strip() for x in line.split('=')]
            print("{} = config.get('{}', '{}')".format(key, section, key))

def get_hyp_inds(selected_hyp_per_agent_matrix, hyp_space, argsort_informativity_per_lexicon=None):
    # get the
    hyp_inds = []
    for hyp in selected_hyp_per_agent_matrix:
        hyp_ind = hyp_space[hyp][1]
        if argsort_informativity_per_lexicon is not None:
            hyp_inds.append(list(argsort_informativity_per_lexicon).index(hyp_ind))
        else:
            hyp_inds.append(hyp_ind)
    return hyp_inds

def normalize(v):
    return np.true_divide(v, v.sum(axis=0,keepdims=1))

def get_lexicon_probs_ambig(lexicon_hyps_all, ind_l, ind_r, sum_val):
    ambig_commA = np.where(lexicon_hyps_all[:,:,ind_l:ind_r].sum(axis=1).sum(axis=1) == sum_val)[0]
    lexicon_probs_commA = np.zeros(lexicon_hyps_all.shape[0])
    lexicon_probs_commA[ambig_commA] = 1
    lexicon_probs_commA = normalize(lexicon_probs_commA)
    return lexicon_probs_commA


def calc_lex_hyp_proportions(hyp_inds, info_dict, null_min_max_info):
    null_info, min_info, max_info = null_min_max_info
    props = [0] * 4
    for hyp_ind in hyp_inds:
        info = info_dict[hyp_ind]
        if info == max_info:
            props[0] += 1
        elif info > min_info:
            props[1] += 1
        elif info == min_info:
            props[2] += 1
        elif null_info <= info < min_info:
            props[3] += 1
        else:
            import pdb; pdb.set_trace()
    return normalize(np.array(props))

def find_best_language(lexicon_hyps, hyp_inds):
    # TODO: might want to think harder and refactor this to have less branches
    def get_all_meanings(meaning1_signals, meaning2_signals):
        return np.unique(np.hstack((meaning1_signals, meaning2_signals)))
    # currently only supports 2 meaning case!
    selected_hyps = lexicon_hyps[hyp_inds]
    selected_hyps_sum = selected_hyps.sum(axis=0)
    max_meaning1, max_meaning2 = selected_hyps_sum.max(axis=1)
    meaning1_signals = np.where(selected_hyps_sum[0] == max_meaning1)[0]
    meaning2_signals = np.where(selected_hyps_sum[1] == max_meaning2)[0]
    arr_eq = np.array_equal(meaning1_signals, meaning2_signals)
    if arr_eq:
        if meaning1_signals.shape[0] == 2:
            all_meanings = meaning1_signals
        elif meaning1_signals.shape[0] == lexicon_hyps.shape[2]:
            # fully ambiguous, return random indices
            all_meanings = np.sort(np.random.choice(meaning1_signals, size=2,replace=False))
        else:
            second_highest1 = np.unique(selected_hyps_sum[0])[-2]
            second_highest2 = np.unique(selected_hyps_sum[1])[-2]
            if second_highest1 >= second_highest2:
                meaning1_signals = np.where(selected_hyps_sum[0] == second_highest1)[0]
            else:
                meaning2_signals = np.where(selected_hyps_sum[1] == second_highest2)[0]
            all_meanings = get_all_meanings(meaning1_signals, meaning2_signals)
    else:
        all_meanings = get_all_meanings(meaning1_signals, meaning2_signals)

    while len(all_meanings) != 2:
        # languages can only be (n x n), so disambiguate
        # first try set diff
        signals1 = np.setdiff1d(meaning1_signals, meaning2_signals)
        signals2 = np.setdiff1d(meaning2_signals, meaning1_signals)
        if signals1.shape[0]:
            meaning1_signals = signals1
        if signals2.shape[0]:
            meaning2_signals = signals2

        # next try deleting lower prob indices
        if np.shape(meaning1_signals) and np.shape(meaning1_signals)[0] > 1:
            min_ind = selected_hyps_sum[1][meaning1_signals].argmin()
            max_ind = selected_hyps_sum[1][meaning1_signals].argmax()
            if min_ind != max_ind:
                meaning1_signals = np.delete(meaning1_signals, max_ind)
        all_meanings = get_all_meanings(meaning1_signals, meaning2_signals)
        if len(all_meanings) == 2: break

        if np.shape(meaning2_signals) and np.shape(meaning2_signals)[0] > 1:
            min_ind = selected_hyps_sum[0][meaning2_signals].argmin()
            max_ind = selected_hyps_sum[0][meaning2_signals].argmax()
            if min_ind != max_ind:
                meaning2_signals = np.delete(meaning2_signals, max_ind)
        all_meanings = get_all_meanings(meaning1_signals, meaning2_signals)
        if len(all_meanings) == 2: break

        if len(all_meanings) != 2:
            # randomly select one of two, to avoid infinite loop!
            if np.shape(meaning1_signals):
                meaning1_signals = np.random.choice(meaning1_signals)
            if np.shape(meaning2_signals):
                if meaning1_signals in meaning2_signals:
                    all_meanings = meaning2_signals
                meaning2_signals = np.random.choice(meaning2_signals)
        all_meanings = get_all_meanings(meaning1_signals, meaning2_signals)

    if len(all_meanings) != 2:
        # last resort
        all_meanings = np.where(selected_hyps_sum == selected_hyps_sum.max())[1]
        if all_meanings.shape > 2:
            all_meanings = np.sort(np.random.choice(all_meanings, size=2,replace=False))
        all_meanings = np.unique(all_meanings)

    return tuple(all_meanings)

# if __name__ == "__main__":
#     shsum = np.array([[ 8., 10.,  9., 10.],
#        [ 4., 10., 10., 10.]])
#     tt = find_best_language(shsum)
