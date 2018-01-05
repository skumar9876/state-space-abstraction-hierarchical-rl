import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle

index = 4
directory = 'experiments_final/experiment_logs/Cheating_Epsilon_Decay_Faster/'

# 6 clusters
sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False']
# sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_True_memory_size_5_pretrain_controller_False']
# sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_True_memory_size_10_pretrain_controller_False']

# 10 clusters
# sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_10_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False']
# sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_10_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_True_memory_size_5_pretrain_controller_False']
# sub_dirs = ['_agent_type_dqn_num_clusters_6_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_False_memory_size_5_pretrain_controller_False', '_agent_type_h_dqn_num_clusters_10_use_extra_travel_penalty_False_use_extra_bit_False_use_controller_dqn_False_use_intrinsic_timeout_False_use_memory_True_memory_size_10_pretrain_controller_False']

color_index = 0
colors = ['r', 'g', 'b']
for sub_dir in sub_dirs:
    mean_rewards = {}
    train_steps = {}
    for i in xrange(4):
        full_dir = directory + sub_dir + '_run_number_' + str(i + 1)

        f = open(full_dir + str('/log.txt'), 'r')
        lines = f.readlines()
        mean_rewards_i = []
        train_steps_i = []
        for j in xrange(min(len(lines), 300)):
            line = lines[j]
            line = line.split(' ')
            mean_rewards_i.append(float(line[2]))

        mean_rewards[i] = mean_rewards_i
        # print mean_rewards[i]
        print len(mean_rewards_i)

        for j in xrange(len(mean_rewards_i)):
            with open(full_dir + '/eval_rewards_' + str(j), 'rb') as data_file:
                dump_dict = pickle.load(data_file)
                train_steps_i.append(dump_dict['train_step'])

        train_steps[i] = train_steps_i

    # interpolate means to 5000 step intervals
    interp_data = [[] for _ in range(4)]
    for task_id in range(4):
        l = 0
        for i in range(0, 700000, 5000):
            while train_steps[task_id][l+1] < i:  # step count on left <= i
                l += 1
            step_l = train_steps[task_id][l]
            step_r = train_steps[task_id][l+1]
            mean_l = np.mean(mean_rewards[task_id][l])
            mean_r = np.mean(mean_rewards[task_id][l+1])
            interp = (i - step_l) * mean_r + (step_r - i) * mean_l
            interp /= (step_r - step_l)
            interp_data[task_id].append((i, interp))

    # print interp_data

    means_0 = [m for (t, m) in interp_data[0]]
    means_1 = [m for (t, m) in interp_data[1]]
    means_2 = [m for (t, m) in interp_data[2]]
    means_3 = [m for (t, m) in interp_data[3]]
    steps = [t for (t, m) in interp_data[0]]
    means_arr = [[means_0[k], means_1[k], means_2[k], means_3[k]] for k in xrange(len(means_0))]
    means = [np.mean(means_arr[k]) for k in xrange(len(means_0))]

    # mean_reward_arrs = [mean_rewards[i] for i in mean_rewards]
    # means = []
    # num_to_plot = 300
    # for i in xrange(num_to_plot):
        # means.append((
        #    mean_rewards[0][i] + mean_rewards[1][i] + mean_rewards[2][i] + mean_rewards[3][i] + mean_rewards[4][i]) / 5.0)
        # means.append((mean_rewards[index][i]))



    # plt.scatter(np.arange(num_to_plot), means, c=colors[color_index])
    # plt.plot(steps, means, c=colors[color_index])
    plt.fill_between(
        steps, [np.percentile(r, 10) for r in means_arr], [np.percentile(
            r, 90) for r in means_arr], facecolor=colors[color_index], alpha=0.2)
    color_index += 1

plt.savefig('plot_' + str(index) + '.png')