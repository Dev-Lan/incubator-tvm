import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


for i in range(12):
    #Works with TVM installed, otherwise load npz
    #file_string = 'task%s.pkl' % str(i)
    #with open(file_string, 'rb') as f:
    #    task, data = pickle.load(f)

    #flat_configs = []
    #flat_features = []
    #flat_results = []

    #include_strings = False

    #shuffled_indices = np.random.permutation(len(data))

    #for index in shuffled_indices:
    #    if data[index].feature is None:
    #        continue
    #    flat_results.append(np.array(data[index].result.costs).mean())

    #    flat_configs.append(data[index].config.get_flatten_feature())
    #    arr = []
    #    for feature in data[index].feature:
    #        first_var = True
    #        if not isinstance(feature, list):
    #            arr.append(feature)
    #        else:
    #            for var in feature:
    #                if include_strings:
    #                    if first_var:
    #                        arr.append(var[0])
    #                        arr.append(var[1].name)
    #                        first_var = False
    #                    else:
    #                        arr.extend(var)
    #                else:
    #                    if first_var:
    #                        first_var = False
    #                    else:
    #                        arr.extend(var[1:])
    #    flat_features.append(arr)

    #train_len = int(len(shuffled_indices)*4/5)
    #x_train_configs = flat_configs[:train_len]
    #x_train = flat_features[:train_len]
    #y_train = flat_results[:train_len]

    #x_test_configs = flat_configs[train_len:]
    #x_test = flat_features[train_len:]
    #y_test = flat_results[train_len:]
    #np.savez('task%i.npz' % i, x_train_configs=x_train_configs, x_train=x_train, y_train=y_train, x_test_configs=x_test_configs, x_test=x_test, y_test=y_test, task=np.array(str(task)))
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

    rf = RandomForestRegressor(n_estimators=10, max_depth=None)
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)

    plt.scatter(y_test, y_predict, marker='.', label='Features')
    rf.fit(x_train_configs, y_train)
    y_predict = rf.predict(x_test_configs)

    plt.scatter(y_test, y_predict, marker='.', label='Config')
    plt.ylabel('Predicted')
    plt.xlabel('Truth')
    plt.legend()
    plt.grid()
    plt.show()
