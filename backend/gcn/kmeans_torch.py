import torch


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        initial_state=None,
        scene_cnts=None
):
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    device = X.device
    initial_state = initial_state.to(device)

    iteration = 0

    dis = pairwise_distance_function(X, initial_state)

    choice_cluster = torch.argmin(dis, dim=1)

    for index in range(num_clusters):
        selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

        selected = torch.index_select(X, 0, selected)

        if len(selected) == 0:
            continue

        w1 = scene_cnts[index]
        weight = w1 + len(selected)

        initial_state[index] = (w1 * initial_state[index] + selected.sum(dim=0)) / weight

    return choice_cluster.to(device), initial_state


def pairwise_distance(data1, data2):
    A = data1.unsqueeze(dim=1)

    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0

    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.unsqueeze(dim=1)

    B = data2.unsqueeze(dim=0)

    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis
