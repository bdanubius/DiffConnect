
import numpy as np
import matplotlib.pyplot as plt
import torch


def omniglot_get_meta_batch(meta_batch_size,n_way,support_k,query_k,is_test=False):

    # imort the module the last moment before use, the first import will trigger the load of the whole dataset to memory
    import omniglot.omniglot_data_singleton
    data = omniglot.omniglot_data_singleton.dataset
    omniglot_shuffled_indicies = omniglot.omniglot_data_singleton.omniglot_shuffled_indicies

    SHUFFLE_CLASSES = True
    AUGMENT_WITH_ROTATION = True

    if SHUFFLE_CLASSES is True:
        train_indicies = omniglot_shuffled_indicies[:1200]
        test_indicies = omniglot_shuffled_indicies[1200:]
    else:
        train_indicies = list(range(1200))
        test_indicies = list(range(1200,data.shape[0]))

    class_indicies = train_indicies
    if is_test is True:
        class_indicies = test_indicies

    support_x = []
    query_x = []
    support_y = []
    query_y = []

    for meta_batch_i in range(meta_batch_size):
        selected_class_indicies = np.random.choice(class_indicies,n_way,replace=False)  

        task_support_x = []
        task_query_x = []
        task_support_y = []
        task_query_y = []

        for class_i_in_batch,class_i in enumerate(selected_class_indicies):

            
            selected_images = np.random.choice(list(range(20)),support_k+query_k,replace=False) # if support_k+query_k = 20, this will be a permutation
            
            class_data = data[class_i,selected_images]

            # Each class can be augmented by rotation, we select the rotation after selecting distinct classes
            # This means we cannot have a task with the same charater with different rotations, which is what we want
            if AUGMENT_WITH_ROTATION is True:
                selected_rotation = np.random.choice([0,1,2,3]) # multiples of 90 degree
                # np.rot90 cannot handle channels, take the one channel, channel 0, and add it back after rotation
                class_data = [np.rot90(class_data[i,0],selected_rotation).reshape(1,28,28) for i in range(len(selected_images))]
                class_data = np.stack(class_data) # we are back to the original shape 

            class_support_x = class_data[:support_k]
            class_query_x = class_data[support_k:]

            class_support_y = np.repeat(class_i_in_batch,support_k)
            class_query_y = np.repeat(class_i_in_batch,query_k)

            task_support_x.append(class_support_x)
            task_query_x.append(class_query_x)
            task_support_y.append(class_support_y)
            task_query_y.append(class_query_y)

        task_support_x = np.stack(task_support_x)
        task_query_x = np.stack(task_query_x)
        task_support_y = np.stack(task_support_y)
        task_query_y = np.stack(task_query_y)

        support_x.append(task_support_x)
        query_x.append(task_query_x)
        support_y.append(task_support_y)
        query_y.append(task_query_y)
    
    support_x = np.stack(support_x)
    query_x = np.stack(query_x)
    support_y = np.stack(support_y)
    query_y = np.stack(query_y)

    # reshape to: meta batch size, batch size, input_size
    support_x = support_x.reshape((meta_batch_size,n_way*support_k,1*28*28))
    query_x = query_x.reshape((meta_batch_size,n_way*query_k,1*28*28))
    support_y = support_y.reshape((meta_batch_size,n_way*support_k))
    query_y = query_y.reshape((meta_batch_size,n_way*query_k))

    return support_x,support_y,query_x,query_y


