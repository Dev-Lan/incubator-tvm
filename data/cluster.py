import numpy as np
from sklearn.cluster import *
import matplotlib.pyplot as plt

for i in range(12):
    data = np.load('task%i.npz' % i)
    x_train_configs = data['x_train_configs']
    x_train = data['x_train']
    y_train = data['y_train']
    x_test_configs = data['x_test_configs']
    x_test = data['x_test']
    y_test = data['y_test']
    task = data['task']
    print('task%i.npz' % i, task)
    print('Number of configs:' , len(y_train)+len(y_test))

    # Number 
    all_clusters = [8,12,16,20]#,25,30,35,40]
    all_samples = [1,2]#,3,5]10,15]
    num_tested = []
    best_time = []
    num_samples = []
    num_clusters = []
    small_num_tested = []
    small_best_time = []
    small_num_samples = []
    small_num_clusters = []

    # Run 5 trials because k-means is random and not consistent?
    for n_trials in range(5):
        print(n_trials)
        for clusters in all_clusters:
            # Do a first clustering on configs, not features, because units
            kmeans = KMeans(n_clusters=clusters).fit(x_train_configs)
            kmeans_train = kmeans.labels_
            # Uncomment to plot k=12 cluster, level 1
            #if (clusters == 12):
            #    plt.scatter(kmeans_train, y_train, marker='.')
            #    plt.xlabel('Cluster')
            #    plt.ylabel('Time')
            #    plt.title('1-level 12-means clustering')
            #    plt.show()
            for n_samples in all_samples:
                random_mins = np.zeros(clusters)
                for ind in range(clusters):
                    random_mins[ind] = np.random.choice(y_train[kmeans_train==ind], size=n_samples).min()
                inds = np.argsort(random_mins)[:2]
                num_tested.append((clusters-1) * n_samples + (kmeans_train==inds[0]).sum())
                best_time.append(y_train[kmeans_train==inds[0]].min())
                num_samples.append(n_samples)
                num_clusters.append(clusters)
                small_x_train = x_train_configs[(kmeans_train==inds[0])]
                small_y_train = y_train[(kmeans_train==inds[0])]
                clusters2 = int(clusters / 2)
                if(len(small_x_train) < clusters):
                    clusters2 = int(len(small_x_train) / 2)
                small_kmeans = KMeans(n_clusters=clusters2).fit(small_x_train)
                small_kmeans_train = small_kmeans.labels_
                random_mins = np.zeros(clusters2)
                for ind in range(clusters2):
                    random_mins[ind] = np.random.choice(small_y_train[small_kmeans_train==ind], size=n_samples).min()
                ind_select = random_mins.argmin()
                small_num_tested.append((clusters-1+clusters2) * n_samples + (small_kmeans_train==ind_select).sum())
                small_best_time.append(small_y_train[small_kmeans_train==ind_select].min())
                small_num_samples.append(n_samples)
                small_num_clusters.append(clusters)
                #Uncomment to plot k=12, 1-sample, level 2
                #if (clusters == 12 and n_samples==1):
                #    plt.scatter(small_kmeans_train, small_y_train, marker='.')
                #    plt.xlabel('Cluster')
                #    plt.ylabel('Time')
                #    plt.title('2-level 12-means, 6-means clustering')
                #    plt.show()


    # Plot the number of samples required vs. performance. 
    best_time = np.array(best_time)
    num_tested = np.array(num_tested)
    num_samples = np.array(num_samples)
    num_clusters = np.array(num_clusters)
    small_best_time = np.array(small_best_time)
    small_num_tested = np.array(small_num_tested)
    small_num_samples = np.array(small_num_samples)
    small_num_clusters = np.array(small_num_clusters)

    # Plot, show number of samples
    for ind in all_samples:
        plt.scatter(best_time[num_samples==ind] , num_tested[num_samples==ind], marker='.', label='%i samples' %ind)
        plt.scatter(small_best_time[small_num_samples==ind] , small_num_tested[small_num_samples==ind], marker='_', label='%i samples (2-level)' %ind)
    plt.xlabel('Time')
    plt.ylabel('Tested Configs')
    plt.legend()
    plt.show()
    # Plot, show clusters 
    for ind in reversed(all_clusters):
        plt.scatter(best_time[num_clusters==ind] , num_tested[num_clusters==ind], s=ind*5, marker='.', label='%i clusters' % ind)
        plt.scatter(small_best_time[small_num_clusters==ind] , small_num_tested[small_num_clusters==ind], s=ind*5, marker='_', label='%i clusters (2-level)' %ind)
    plt.xlabel('Time')
    plt.ylabel('Tested Configs')
    plt.legend()
    plt.show()

