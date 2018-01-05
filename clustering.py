"""
Implmentation of functions which form a clustering over the state space and 
classify an input state into one of the previously formed clusters.

@author: Saurabh Kumar
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def make_clusters(env_name, n_clusters):
    env = gym.make(env_name)
    env.reset()
    VALID_ACTIONS = list(range(env.action_space.n))

    data = []

    for episode in xrange(1000):
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            step_count += 1
            action = random.randint(0, len(VALID_ACTIONS) - 1)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            state_with_reward = state[:]
            state_with_reward = np.append(state_with_reward, reward)
            data.append(state_with_reward)

    data = np.array(data)
    x_pos_normalized = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
    velocity_normalized = (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1])
    # reward_normalized = (data[:, 2] - np.mean(data[:, 2])) / np.std(data[:, 2])
    # data_normalized = zip(x_pos_normalized, velocity_normalized, reward_normalized)
    data_normalized = zip(x_pos_normalized, velocity_normalized)

    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data_normalized)

    means = [np.mean(data[:, 0]), np.mean(data[:, 1])]
    stds = [np.std(data[:, 0]), np.std(data[:, 1])]

    cluster_centers = estimator.cluster_centers_[:,0:2]
    for i in xrange(len(cluster_centers)):
        cluster_centers[i][0] = cluster_centers[i][0] * stds[0] + means[0]
        cluster_centers[i][1] = cluster_centers[i][1] * stds[1] + means[1]

    labels = estimator.labels_
    labels = labels.astype(np.int32)
    colors = ['red', 'green', 'blue', 'orange',
    'yellow', 'magenta', 'black',
    'purple', 'brown', 'white']

    fig, ax = plt.subplots()
    for i in xrange(n_clusters):
        label = i
        color = colors[i%len(colors)]
        indices_of_labels = np.where(labels==label)
        ax.scatter(data[indices_of_labels,0][0], data[indices_of_labels,1][0], c=color,
            label=int(label), alpha=0.5)

    ax.legend()
    plt.xlabel('X Position')
    plt.ylabel('Velocity')

    try:
        os.stat('clusters_' + str(n_clusters))
    except:
        os.mkdir('clusters_' + str(n_clusters))

    plt.savefig('clusters_' + str(n_clusters) + '/Clusters.png')

    returned_data = zip(data[:, 0], data[:, 1])

    with open('clusters_' + str(n_clusters) + '/data', 'w') as data_file:
        pickle.dump(returned_data, data_file)
    with open('clusters_' + str(n_clusters) + '/labels', 'w') as labels_file:
        pickle.dump(labels, labels_file)
    with open('clusters_' + str(n_clusters) + '/cluster_centers', 'w') as cluster_centers_file:
        pickle.dump(cluster_centers, cluster_centers_file)

    return returned_data, labels, cluster_centers


def get_cluster_fn(env_name='MountainCar-v0', n_clusters=10, extra_bit=True, load_from_dir=True):
    if load_from_dir:
        with open('clusters_' + str(n_clusters) + '/data', 'rb') as data_file:
            data = pickle.load(data_file)
        with open('clusters_' + str(n_clusters) + '/labels', 'rb') as labels_file:
            labels = pickle.load(labels_file)
        with open('clusters_' + str(n_clusters) + '/cluster_centers', 'rb') as cluster_centers_file:
            cluster_centers = pickle.load(cluster_centers_file)

    else:
        data, labels, cluster_centers = make_clusters(env_name, n_clusters)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(data, labels)
    # Create one-hot representation of the clusters.
    clusters_one_hot = [np.zeros(n_clusters) for i in xrange(n_clusters)]
    for i in xrange(len(clusters_one_hot)):
        clusters_one_hot[i][i] = 1

    ratio = 0.5

    def check_cluster(data_point, cluster_index, original_point=None):
        # print "Check cluster function:"
        # print cluster_index
        if not extra_bit or original_point is None:
            predicted_cluster_index = neigh.predict(data_point)[0]
            data_point = np.squeeze(data_point)
            # Cheating for the goal cluster area!
            if data_point[0] >= 0.5:
                predicted_cluster_index = 5
            if data_point[0] < 0.5 and predicted_cluster_index == 5:
                predicted_cluster_index = 3

	    return cluster_index == predicted_cluster_index
        else:
            distance_to_boundary = euclidean_distance(data_point, original_point)
            distance_to_center = euclidean_distance(data_point, cluster_centers[cluster_index])
            return np.float(distance_to_center) / np.maximum(distance_to_boundary, np.exp(-10)) <= ratio


    def identify_cluster(data_point, original_point):
        cluster_index = neigh.predict(data_point)[0]
        data_point = np.squeeze(data_point)
        # Cheating for the goal cluster area!
        if data_point[0] >= 0.5:
            cluster_index = 5
        if data_point[0] < 0.5 and cluster_index == 5:
            cluster_index = 3

        if extra_bit:
            cluster_one_hot = np.zeros(n_clusters + 1)
        else:
            cluster_one_hot = np.zeros(n_clusters)
        cluster_one_hot[cluster_index] = 1

        if extra_bit:
            # Add bit that represents whether agent is on boundary or in center of cluster
            if original_point is not None:
                distance_to_boundary = euclidean_distance(data_point, original_point)
                distance_to_center = euclidean_distance(data_point, cluster_centers[cluster_index])
                if np.float(distance_to_center) / np.maximum(distance_to_boundary, np.exp(-10)) <= ratio:
                    cluster_one_hot[-1] = 1

        return cluster_one_hot

    return identify_cluster, check_cluster, n_clusters, np.array(clusters_one_hot)


def euclidean_distance(point1, point2):
    point1 = np.squeeze(point1)
    point2 = np.squeeze(point2)
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
