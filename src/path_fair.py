"""
    This is the code implementation of PathFair

    This code is based on NeuronFair:
    @inproceedings{Zheng2021NeuronFair,
      title={NeuronFair: Interpretable White-Box Fairness Testing through Biased Neuron Identification},
      author={Haibin Zheng, Zhiqing Chen, Tianyu Du, Xuhong Zhang, Yao Cheng, Shouling Ji, Jingyi Wang, Yue Yu, and Jinyin Chen*},
      booktitle={The 44th International Conference on Software Engineering (ICSE 2022), May 21-29, 2022, Pittsburgh, PA, USA},
      pages = {1--13},
      year={2022}
    }
    GitHub repository: https://github.com/haibinzheng/NeuronFair

    The calculation of activation path comes from FairNeuron:
    @INPROCEEDINGS{9793993,
      author={Gao, Xuanqi and Zhai, Juan and Ma, Shiqing and Shen, Chao and Chen, Yufei and Wang, Qian},
      booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE)},
      title={Fairneuron: Improving Deep Neural Network Fairness with Adversary Games on Selective Neurons},
      year={2022},
      pages={921-933},
      doi={10.1145/3510003.3510087}}
    GitHub repository:https://github.com/Antimony5292/FairNeuron

    According to NeuronFair's Apache License 2.0, the changed contents of this file are presented
    as follows:
    1. Change the file name from 'dnn_nf' to 'path_fair'
    2. Add judgement to the import of datasets in order to increase the performance of program
    3. Add implementation of activation path calculation
    4. Change the computation of ActDiff to match NeuronFair paper
    5. Add momentum to match NeuronFair paper
    6. Fix the calculation of gradient sign
    7. Fix the calculation of attribute sampling possibility
    8. Fix the perturbation operation in local generation stage
    9. Extract arguments of functions and use argparse to get parameters of the program
"""
import argparse
import os
import sys
import time
import copy
import numpy as np
from scipy.optimize import basinhopping

# import tensorflow as tf
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, '../'))
from tensorflow_version import tf
from nf_model.dnn_models import dnn
from utils.utils_tf import model_prediction, model_argmax, model_loss
from src.nf_utils import cluster

olderr = np.seterr(all='ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    t = t.astype('int')
    label = model_argmax(sess, x, preds, np.array([t]))
    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1):
        if val != t[sens - 1]:
            tnew = copy.deepcopy(t)
            tnew[sens - 1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return True
    return False


def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)


def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input


class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, x, nx, x_grad, nx_grad, n_value, sens_param, input_shape, conf, local_decay=0.05):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param x: input placeholder for x
        :param nx: input placeholder for nx (sensitive attributes of nx and x are different)
        :param x_grad: the gradient graph for x
        :param nx_grad: the gradient graph for nx
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        :param local_decay: the decay value of momentum in local generation stage
        """
        self.sess = sess
        self.grad = x_grad
        self.ngrad = nx_grad
        self.x = x
        self.nx = nx
        self.n_value = n_value
        self.input_shape = input_shape
        self.sens_param = sens_param
        self.conf = conf
        self.pre_grad = 0
        self.pre_ngrad = 0
        self.local_decay = local_decay

    def softmax(self, m):
        probs = np.exp(m - np.max(m))
        probs /= np.sum(probs)
        return probs

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """
        n_x = x.copy()
        n_x[self.sens_param - 1] = self.n_value
        ind_grad, n_ind_grad = self.sess.run([self.grad, self.ngrad],
                                             feed_dict={self.x: np.array([x]), self.nx: np.array([n_x])})
        ind_grad = self.pre_grad * self.local_decay + ind_grad
        n_ind_grad = self.pre_ngrad * self.local_decay + n_ind_grad
        self.pre_grad = ind_grad
        self.pre_ngrad = n_ind_grad
        signs = np.sign(ind_grad + n_ind_grad)

        if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and np.zeros(self.input_shape).tolist() == \
                n_ind_grad[0].tolist():
            probs = 1.0 / (self.input_shape - 1) * np.ones(self.input_shape)
            probs[self.sens_param - 1] = 0
        else:
            grad_sum = 1.0 / (abs(ind_grad[0] + n_ind_grad[0]))
            grad_sum[self.sens_param - 1] = 0
            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()
        try:
            index = np.random.choice(range(self.input_shape), p=probs)
        except:
            index = 0
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0
        x = clip(x + signs[0][index] * local_cal_grad, self.conf).astype("int")
        return x


def gradient_graph_neuron(x, nx, weights, hidden, nhidden):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param nx: the input placeholder
    :param weights: the weight of most biasd layer neurons
    :param hidden: placeholder of neurons in most biased layer
    :param nhidden: placeholder of neurons in most biased layer
    :return: the gradient graph
    """
    tf_weights = tf.constant(weights, dtype=tf.float32)
    x_loss = model_loss(nhidden * tf_weights, hidden * tf_weights, mean=False)
    nx_loss = model_loss(hidden * tf_weights, nhidden * tf_weights, mean=False)
    x_gradients = tf.gradients(x_loss, x)[0]
    nx_gradients = tf.gradients(nx_loss, nx)[0]
    return x_gradients, nx_gradients


crash_cnt = 0
suc_crash_cnt = 0
eval_times = 0


def dnn_fair_testing(dataset, sensitive_param, model_path, cluster_num, max_global, max_local, max_iter, ReLU_name,
                     data_preproc, perturbation_size=1, uni_seed=3607, local_decay=0.05,
                     global_dacay=0.1,
                     bias_alpha=0.6375, **kwargs):
    start_time = time.time()
    """
    The implementation of NF and PathFair
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    :param ReLU_name: the name of bias layer of dnn model
    :param data_preproc: the pre-processing method, one of random/cluster/actpath/actpath_front,
         cluster corresponds to Pcluster in paper, actpath to Pact_top, and actpath_front to 
         Pact_sample
    :param perturbation_size: the perturbation step size in local and global generation
    :param uni_seed: the random seed
    :param local_decay: the decay value of momentum in local generation stage
    :param global_decay: the decay value of momentum in global generation stage
    :param bias_alpha: the percentage of biased neurons in the most biased layer
    """
    data_config = {}
    if dataset == 'census':
        from nf_data.census import census_data
        from utils.config import census
        data_tuple = census_data()
        data_config[dataset] = census
    elif dataset == 'credit':
        from nf_data.credit import credit_data
        from utils.config import credit
        data_tuple = credit_data()
        data_config[dataset] = credit
    elif dataset == 'bank':
        from nf_data.bank import bank_data
        from utils.config import bank
        data_tuple = bank_data()
        data_config[dataset] = bank
    elif dataset == 'compas':
        from nf_data.compas import compas_data
        from utils.config import compas
        data_tuple = compas_data()
        data_config[dataset] = compas
    elif dataset == 'meps':
        from nf_data.meps import meps_data
        from utils.config import meps
        data_tuple = meps_data()
        data_config[dataset] = meps

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data_tuple

    def get_weights(X, sensitive_param, sess, x, nx, x_hidden, nx_hidden, alpha=0.7):
        nX = copy.copy(X)
        senss = data_config[dataset].input_bounds[sensitive_param - 1]
        eq = np.array(nX[:, sensitive_param - 1] == senss[0]).astype(np.int)
        neq = -eq + 1
        nX[:, sensitive_param - 1] = eq * senss[-1] + neq * senss[0]
        sa, nsa = sess.run([x_hidden, nx_hidden], feed_dict={x: X, nx: nX})
        diff_val = np.abs(sa - nsa)
        sf = np.mean(diff_val, axis=0)
        num = 0 if int(alpha * len(sf)) - 1 < 0 else int(alpha * len(sf)) - 1
        ti = np.argsort(sf)[len(sf) - num - 1]
        alpha = sf[ti]
        weights = np.array(sf >= alpha).astype(np.int)
        return weights

    tf.set_random_seed(uni_seed if uni_seed else 2020)
    if uni_seed:
        np.random.seed(uni_seed)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sd = 0
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        x = tf.placeholder(tf.float32, shape=input_shape)
        nx = tf.placeholder(tf.float32, shape=input_shape)
        model, linear_weights = dnn(input_shape, nb_classes, get_weights=True)
        preds = model(x)
        x_hidden = model.get_layer(x, ReLU_name)
        nx_hidden = model.get_layer(nx, ReLU_name)
        x_all_hidden, act_dict = model.get_all_activation(x, with_dict=True, only_hidden=False)
        linear_dict = {}
        for i in act_dict:
            if 'Linear' in i:
                linear_dict[i] = act_dict[i]
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        if data_preproc == 'actpath' or data_preproc == 'actpath_front':
            """
            FairNeuron activation path calculation.
            """

            def get_active_neurons_and_paras(_sess, _sample, linears_dict, linears_weight):
                result = _sess.run([linears_dict[_i] for _i in linears_dict] + linears_weight,
                                   feed_dict={x: _sample, nx: _sample})
                return result[:len(linears_dict)], result[len(linears_dict):]

            def get_contrib4(paras, neurons, layer_num=3):
                contrib_list = []
                for _i in range(layer_num):
                    contrib = neurons[_i].T * paras[_i + 1]
                    contrib_list.append(contrib)
                return contrib_list

            def get_path_set4(net, _sample, linears_dict, linears_weight, GAMMA=0.9, layer_num=None):
                path_set = set()
                neurons, paras = get_active_neurons_and_paras(net, _sample, linears_dict, linears_weight)
                if not layer_num:
                    layer_num = len(neurons)
                contrib_list = get_contrib4(paras, neurons, layer_num=layer_num - 1)
                active_neuron_indice = [[] for __i in range(layer_num)]
                active_neuron_indice[layer_num - 1].append(np.argmax(neurons[layer_num - 1]))
                for _i in range(layer_num - 1):
                    L = layer_num - 1 - _i
                    for j in active_neuron_indice[L]:
                        pre_neurons = contrib_list[L - 1][:, j]
                        s = np.argsort(-pre_neurons)
                        _sum = 0
                        for k in range(len(contrib_list[L - 1][j])):
                            _sum += pre_neurons[s[k]]
                            active_neuron_indice[L - 1].append(s[k])
                            path_set.add((L, s[k], j))
                            if _sum >= GAMMA * neurons[L][0, j]:
                                break
                return frozenset(path_set)

            path_idx_dict = {}
            path_cnt_dict = {}
            for _idx in range(X.shape[0]):
                _i = X[_idx]
                cur_path_set = get_path_set4(sess, _i[np.newaxis, :], linear_dict, linear_weights)
                if cur_path_set in path_idx_dict:
                    path_idx_dict[cur_path_set].add(_idx)
                else:
                    path_idx_dict[cur_path_set] = {_idx}
                path_cnt_dict[cur_path_set] = path_cnt_dict[cur_path_set] + 1 if cur_path_set in path_cnt_dict else 1

            path_cnt_list = [(path_cnt_dict[i], i) for i in path_cnt_dict]
            path_cnt_list.sort()
            path_sorted_list = []
            for _i in path_cnt_list:
                cur_path = _i[1]
                path_sorted_list += list(path_idx_dict[cur_path])

        weights = get_weights(X, sensitive_param, sess, x, nx, x_hidden, nx_hidden, alpha=bias_alpha)
        x_grad, nx_grad = gradient_graph_neuron(x, nx, weights, x_hidden,
                                                nx_hidden)
        if data_preproc == 'cluster':
            clf = cluster(dataset, cluster_num, random_seed=uni_seed if uni_seed else 2019)

            clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]
            inputs = seed_test_input(clusters, min(max_global, len(X)))
        # store the result of fairness testing
        tot_inputs = set()
        global_disc_inputs = set()
        global_disc_inputs_list = []
        local_disc_inputs = set()
        local_disc_inputs_list = []
        value_list = []
        suc_idx = []

        # --- my code
        prev_iter_cnt = 0
        iter_cnt = 0

        def evaluate_local(inp):
            """
            Evaluate whether the test input after local perturbation is an individual discriminatory instance
            :param inp: test input
            :return: whether it is an individual discriminatory instance
            """
            global eval_times
            eval_times += 1
            result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)

            temp = copy.deepcopy(inp.astype('int').tolist())

            temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
            tuple_temp = tuple(temp)
            if tuple_temp in tot_inputs:
                global crash_cnt
                crash_cnt += 1
            tot_inputs.add(tuple_temp)
            if result and ((tuple_temp in global_disc_inputs) or (tuple_temp in local_disc_inputs)):
                global suc_crash_cnt
                suc_crash_cnt += 1
            if result and (tuple_temp not in global_disc_inputs) and (tuple_temp not in local_disc_inputs):
                local_disc_inputs.add(tuple_temp)
                local_disc_inputs_list.append(temp)
            return not result

        global_input_size = min(max_global, len(X))
        print(f"Global input num:{global_input_size}")

        # select the seed input for fairness testing
        if data_preproc == 'actpath':
            X = X[path_sorted_list]
        elif data_preproc == 'actpath_front':
            X = X[path_sorted_list]
            if 'sample_rate' in kwargs:
                num_data = len(X)
                sample_num = global_input_size
                step_rate = kwargs['sample_rate']
                step_size = 1
                if sample_num > 1:
                    step_size = max(int(((num_data - 1) / (sample_num - 1)) * step_rate), 1)
                X = X[::step_size, ]
            else:
                raise Exception('Data preprocess method "actpath_front" needs sample_rate')
        elif data_preproc == 'random':
            np.random.shuffle(X)

        for num in range(global_input_size):
            if data_preproc == 'cluster':
                index = inputs[num]
            elif data_preproc == 'actpath' or data_preproc == 'random' or data_preproc == 'actpath_front':
                index = num
            else:
                raise Exception('Unknown data preprocess method')
            sample = X[index:index + 1]

            memory1 = sample[0] * 0
            # [1] * sample.shape[1]
            memory2 = sample[0] * 0 + 1
            # [-1] * sample.shape[1]
            memory3 = sample[0] * 0 - 1
            # start global perturbation
            pre_s_grad = pre_n_grad = 0
            for iter in range(max_iter + 1):
                iter_cnt += 1
                probs = model_prediction(sess, x, preds, sample)[0]
                label = np.argmax(probs)
                prob = probs[label]
                max_diff = 0
                n_value = -1
                for i in range(data_config[dataset].input_bounds[sensitive_param - 1][0],
                               data_config[dataset].input_bounds[sensitive_param - 1][1] + 1):
                    if i != sample[0][sensitive_param - 1]:
                        n_sample = sample.copy()
                        n_sample[0][sensitive_param - 1] = i
                        n_probs = model_prediction(sess, x, preds, n_sample)[0]
                        n_label = np.argmax(n_probs)
                        n_prob = n_probs[n_label]
                        if label != n_label:
                            n_value = i
                            break
                        else:
                            prob_diff = abs(prob - n_prob)
                            if prob_diff > max_diff:
                                max_diff = prob_diff
                                n_value = i

                temp = copy.deepcopy(sample[0].astype('int').tolist())

                temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
                # if get an individual discriminatory instance
                if label != n_label and (tuple(temp) not in global_disc_inputs) and (
                        tuple(temp) not in local_disc_inputs):
                    global_disc_inputs_list.append(temp)
                    global_disc_inputs.add(tuple(temp))
                    value_list.append([sample[0, sensitive_param - 1], n_value])
                    suc_idx.append(index)
                    # start local perturbation
                    minimizer = {"method": "L-BFGS-B"}
                    local_perturbation = Local_Perturbation(sess, x, nx, x_grad, nx_grad, n_value, sensitive_param,
                                                            input_shape[1], data_config[dataset],
                                                            local_decay=local_decay)
                    basinhopping(evaluate_local, sample, stepsize=1.0, take_step=local_perturbation,
                                 minimizer_kwargs=minimizer,
                                 niter=max_local)
                    global crash_cnt
                    global eval_times
                    cur_time = time.time()
                    print(f"current time cost:{cur_time - start_time}")
                    print(f"eval times:{eval_times},crash cnt:{crash_cnt},successful crash cnt:{suc_crash_cnt}")
                    print(f"crash rate:{crash_cnt / eval_times},successful crash rate:{suc_crash_cnt / eval_times}")
                    print('iter cnt inc:', iter_cnt - prev_iter_cnt, '| total iter cnt:', iter_cnt)
                    prev_iter_cnt = iter_cnt

                    print(len(tot_inputs), num, len(local_disc_inputs),
                          "Percentage discriminatory inputs of local search- " + str(
                              float(len(local_disc_inputs)) / float(len(tot_inputs) + 1) * 100))
                    print("Total cal cnt:", (len(tot_inputs) + iter_cnt + 1), "Total calculate effectiveness:",
                          (len(local_disc_inputs) + len(global_disc_inputs)) / (len(tot_inputs) + iter_cnt + 1))
                    print('-' * 120)
                    break

                n_sample[0][sensitive_param - 1] = n_value

                s_grad, n_grad = sess.run([x_grad, nx_grad],
                                          feed_dict={x: sample, nx: n_sample})
                s_grad = pre_s_grad * global_dacay + s_grad
                n_grad = pre_n_grad * global_dacay + n_grad
                sn_grad = np.sign(s_grad + n_grad)
                s_grad = np.sign(s_grad)
                n_grad = np.sign(n_grad)

                if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                    g_diff = n_grad[0]
                elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                    g_diff = s_grad[0]
                else:
                    g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)

                g_diff[sensitive_param - 1] = 0
                if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                    g_diff = sn_grad[0]
                    g_diff[sensitive_param - 1] = 0
                if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist() or np.array(
                        memory1[0]).tolist() == np.array(memory3[0]).tolist():
                    np.random.seed(seed=((uni_seed if uni_seed else 2020) + sd))
                    sd += 1
                    delta = perturbation_size
                    s_grad[0] = np.random.randint(-delta, delta + 1, (np.shape(s_grad[0])))
                g_diff = sn_grad[0]
                g_diff[sensitive_param - 1] = 0
                cal_grad = g_diff

                memory1 = memory2
                memory2 = memory3
                memory3 = cal_grad
                perturb_value = perturbation_size * cal_grad
                perturb_value = perturb_value + sample[0]
                sample[0] = clip(perturb_value, data_config[dataset]).astype("int")
                if iter == max_iter:
                    break

        end_time = time.time()
        print(f"Total time cost:{end_time - start_time}")
        print("Total Inputs are " + str(len(tot_inputs)))
        print("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)))
        print("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)))


def resolve_config_args(cfg_path, type_cast=True):
    args = {}
    with open(cfg_path, 'r', encoding='utf-8') as cfg_file:
        lines = cfg_file.readlines()
        for j in lines:
            if '=' in j:
                arr = j.rstrip().split('=')
                if type_cast:
                    try:
                        args[arr[0]] = int(arr[1])
                    except:
                        try:
                            args[arr[0]] = float(arr[1])
                        except:
                            args[arr[0]] = arr[1]
                else:
                    args[arr[0]] = arr[1]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description='PathFair program arguments', add_help=False)
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(cur_path, '..', 'configs', 'census_gender_actpath_front.cfg'),
                        help='Path of config file to run the PathFair program, absolute path is recommended')
    parsed_args, _ = parser.parse_known_args()
    pathfair_args = resolve_config_args(parsed_args.config_path)
    print(f"Experiment name:{pathfair_args['exp_name']}")
    print(f"Experiment arguments:{pathfair_args}")
    dnn_fair_testing(**pathfair_args)


if __name__ == '__main__':
    tf.app.run()
