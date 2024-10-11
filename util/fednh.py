import torch
import torch.nn.functional as F
import copy

def fednh(local_protos_list, central_node):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    init_global_prototypes = copy.deepcopy(central_node.prototype.data)
    global_prototypes = agg_protos_label

    # global: dict --- > tensor
    global_pro_list = []
    proto_label = global_prototypes.keys()
    for i in range(central_node.num_classes):
        if i in proto_label:
            value = global_prototypes[i][0]
            # global_pro_list.append(global_prototypes[i][0])
        else:
            value = torch.zeros(1024)
        global_pro_list.append(value)
    global_pro_tensor = torch.stack(global_pro_list, dim=0)

    avg_prototype = F.normalize(global_pro_tensor, dim=1)
    # update prototype with moving average
    weight = 0.9
    global_proto = weight * init_global_prototypes + (1 - weight) * avg_prototype
    # print('agg weight:', weight)
    # normalize prototype again
    global_proto = F.normalize(global_proto, dim=1)
    return global_proto