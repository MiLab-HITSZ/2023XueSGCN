import numpy as np




def aggregate_bn_data(bn_tensors, degrees=None):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
    client_nums = len(bn_tensors)
    layer_nums = len(bn_tensors[0]) // 2
    bn_data = []
    
    for i in range(layer_nums):
        mean_idx = i * 2
        mean_var_dim = len(bn_tensors[0][mean_idx])
        mean = np.zeros(mean_var_dim)
        
        for idx in range(client_nums):
            
            client_mean = bn_tensors[idx][mean_idx]
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            mean += client_mean * client_weight
        mean /= total_weight
        bn_data.append(mean)
        
        var_idx = mean_idx + 1
        var = np.zeros(mean_var_dim)
        for idx in range(client_nums):
            client_mean = bn_tensors[idx][mean_idx]
            client_var = bn_tensors[idx][var_idx]
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            var += (client_var + client_mean ** 2 - mean ** 2) * client_weight
        var /= total_weight
        bn_data.append(var)
    return bn_data



def aggregate_by_labels(tensors, degrees):
    
    
    
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    
    for i in range(len(tensors)):
        for j, tensor in enumerate(tensors[i]):
            
            if j == len(tensors[i]) - 2 or j == len(tensors[i]) - 1:
                
                for k in range(len(tensor)):
                    
                    
                    if degrees_sum[k] == 0:
                        tensor[k] *= degrees[i][-1]
                        tensor[k] /= degrees_sum[-1]
                    else:
                        tensor[k] *= degrees[i][k]
                        tensor[k] /= degrees_sum[k]
                    if i != 0:
                        tensors[0][j][k] += tensor[k]
            else:
                tensor *= degrees[i][-1]
                tensor /= degrees_sum[-1]
                if i != 0:
                    tensors[0][j] += tensor
    
    return tensors[0]


def aggregate_whole_model(tensors, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
    for i in range(len(tensors)):
        client_weight = degrees[i] if degrees.ndim == 1 else degrees[i][-1]
        for j, tensor in enumerate(tensors[i]):
            tensor *= client_weight
            tensor /= total_weight
            if i != 0:
                tensors[0][j] += tensor
    return tensors[0]


def aggregate_relation_matrix(relation_matrices, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    client_nums = len(relation_matrices)
    relation_matrix = np.zeros_like(relation_matrices[0])
    for i in range(client_nums):
        relation_matrix += relation_matrices[i] * degrees[i][-1] / degrees_sum[-1]
    return relation_matrix


def aggregate_scene_adjs_with_cnts(scene_infos):
    num_clients = len(scene_infos)
    num_scenes = len(scene_infos[0][0])
    num_labels = len(scene_infos[0][1][0])
    fixed_adjs = np.zeros((num_clients, num_scenes, num_labels, num_labels))
    names = [scene_infos[i][3] for i in range(num_clients)]
    for i in range(num_clients):
        linear_i = scene_infos[i][0]
        for k, scene_k in enumerate(linear_i):
            
            coefficients = [None] * num_clients
            total_cnt = scene_infos[i][2][k]  
            cosine_similarities = np.zeros(num_scenes)
            
            for j in range(num_clients):
                if j == i:
                    continue
                linear_j = scene_infos[j][0]
                for l, scene_l in enumerate(linear_j):
                    dot_product = np.dot(scene_k, scene_l)
                    norm_vector1 = np.linalg.norm(scene_k)
                    norm_vector2 = np.linalg.norm(scene_l)
                    cosine_similarities[l] = dot_product / (norm_vector1 * norm_vector2)
                
                max_scene_id = cosine_similarities.argmax()  
                max_scene_similarity = cosine_similarities[max_scene_id]
                
                if max_scene_similarity < 0:
                    continue
                coefficients[j] = [max_scene_id, max_scene_similarity]
                total_cnt += scene_infos[j][2][max_scene_id]
            
            
            other_weights = 0
            agg_scene_adj = np.zeros((num_labels, num_labels))
            for j in range(num_clients):
                if coefficients[j] is not None:
                    
                    scene_id, weight = coefficients[j]
                    coefficients[j][1] = weight * scene_infos[j][2][scene_id] / total_cnt
                    other_weights += coefficients[j][1]
                    
                    agg_scene_adj += scene_infos[j][1][scene_id] * coefficients[j][1]
            self_coefficient = 1 - other_weights
            agg_scene_adj += self_coefficient * scene_infos[i][1][k]
            
            fixed_adjs[i][k] = agg_scene_adj
    return (names, fixed_adjs)







def aggregate_scene_adjs(scene_infos):
    num_clients = len(scene_infos)
    
    num_scenes_list = [len(scene_infos[i][0]) for i in range(num_clients)]
    num_labels = len(scene_infos[0][1][0])
    
    adj_list = [np.zeros((num_scenes, num_labels, num_labels)) for num_scenes in num_scenes_list]
    names = [scene_infos[i][3] for i in range(num_clients)]
    for i in range(num_clients):
        linear_i = scene_infos[i][0]
        for k, scene_k in enumerate(linear_i):
            
            coefficients = [None] * num_clients
            
            for j in range(num_clients):
                if j == i:
                    continue
                linear_j = scene_infos[j][0]
                
                cosine_similarities = np.zeros(num_scenes_list[j])
                for l, scene_l in enumerate(linear_j):
                    dot_product = np.dot(scene_k, scene_l)
                    norm_vector1 = np.linalg.norm(scene_k)
                    norm_vector2 = np.linalg.norm(scene_l)
                    cosine_similarities[l] = dot_product / (norm_vector1 * norm_vector2)
                
                max_scene_id = cosine_similarities.argmax()  
                max_scene_similarity = cosine_similarities[max_scene_id]
                
                if max_scene_similarity < 0:
                    continue
                coefficients[j] = [max_scene_id, max_scene_similarity]
            
            agg_scene_adj = np.zeros((num_labels, num_labels))
            
            total_weight = 1
            for j in range(num_clients):
                if coefficients[j] is not None:
                    
                    scene_id, weight = coefficients[j]
                    total_weight += coefficients[j][1]
                    agg_scene_adj += scene_infos[j][1][scene_id] * coefficients[j][1]
            agg_scene_adj += scene_infos[i][1][k]
            agg_scene_adj /= total_weight
            
            adj_list[i][k] = agg_scene_adj
    return names, adj_list
