import numpy as np
import os
import torch
import torchvision.transforms as transforms
from util.fedavg import FedAvg_Rod
import sys

import copy
import warnings
import time

warnings.filterwarnings("ignore")

from torch.utils.data.dataset import Dataset
from PIL import Image
from glob import glob
from itertools import chain
from Model.ClassificationM import MyClassification
from options import args_parser
from torch.utils.data import dataloader
from multiprocessing.reduction import ForkingPickler
from tqdm import tqdm
from util.sampling import non_iid_dirichlet_sampling
from chest.validate import globaltest15
from torch import nn
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# _gpu_count=1

class LocalUpdate(object):
    def __init__(self, args, train_dataset, device='cpu', net=None, lr=None):
        self.args = args
        self.ldr_train = train_dataset
        self.device = device
        self.losses = torch.nn.CrossEntropyLoss()
        self.net = net
        self.net.to(self.device)
        if lr is None:
            self.lr = self.args.lr
        else:
            self.lr = lr
        self.backbone_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.args.momentum)

    def update_weights(self, epoch):
        self.net.train()
        epoch_loss = []
        gradient_tail_vec = torch.zeros(self.net.fc.weight[9:].shape).to(self.device)

        for iter in range(epoch):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.squeeze()
                labels = labels.long()
                self.net.zero_grad()

                feat = self.net(images)
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                loss = self.losses(feat, labels)
                loss.backward()
                gradient_tail_vec += self.net.fc.weight.grad[9:] * self.lr
                self.backbone_optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        acc = 0.
        return self.net.cpu(), sum(epoch_loss) / len(epoch_loss), gradient_tail_vec, acc

class xrayDataset(Dataset):
    def __init__(self, dataset, is_trainset, idxs, classes):
        # self.dataframe = dataframe
        if idxs != None:
            self.idxs = list(idxs)
        self.dataset= dataset
        self.classes = classes
        self.is_train = is_trainset
        # self.images = images

    def __len__(self):
        if self.is_train:
            return len(self.idxs)
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        if self.is_train:
            image, label = self.dataset[self.idxs[item]]
        else:
            image, label = self.dataset[item]
        return (image, label)


def local_train(q, g, args, _train, idx, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    local = LocalUpdate(args=args, train_dataset=_train, device=device, net=g)
    backbone_w_local, loss_local, _vec, _acc = local.update_weights(epoch=args.local_ep)
    print('Epoch {} \t user:{} \t loss={}'.format(args.rounds, idx, loss_local))
    q.put([backbone_w_local.state_dict(), loss_local, _vec.cpu(), _acc, idx])
    return


if __name__ == '__main__':
    default_collate_func = dataloader.default_collate

    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)

    setattr(dataloader, 'default_collate', default_collate_override)

    for t in torch._storage_classes:
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]

    args = args_parser()
    print(args)

    classes = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax',
               'Consolidation', 'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia',
               'Hernia']

    print("start load train and test images")
    start_time = time.time()
    train_dateset = torch.load('./chest_x_ray/train_15c_dataset{}_{}_features.pt'.format(args.chest_data, args.model))
    print('train dataset size:{}'.format(len(train_dateset)))
    labels = np.asarray([l for _, l in train_dateset])
    test_dateset = torch.load('./chest_x_ray/test_15c_dataset500_balance_{}_features.pt'.format(args.model))
    print('test dataset size:{}'.format(len(test_dateset)))
    end_time = time.time()
    execution_time = end_time - start_time
    print("finished load train and test images . {}s.".format(execution_time))

    classes_num = len(classes)

    dict_users = non_iid_dirichlet_sampling(labels, args.num_classes, args.non_iid_prob_class, args.num_users,
                                            args.seed, args.alpha_dirichlet)
    alist = np.array([[np.sum(labels[list(dict_users[i])] == j) for j in range(classes_num)] for i in range(args.num_users)])
    print("training set distribution:")
    print(alist)
    clients_sizes = [len(dict_users[i]) for i in range(args.num_users)]
    print("clients_sizes:{}".format(clients_sizes))

    users_dataset_train = []
    for idx in range(args.num_users):
        users_dataset_train.append(
            dataloader.DataLoader(xrayDataset(train_dateset, True, dict_users[idx], classes), batch_size=args.local_bs,
                                  shuffle=False, pin_memory=True, num_workers=4))

    test_loader = dataloader.DataLoader(xrayDataset(test_dateset, False, None, classes), batch_size=128, shuffle=False,
                                        pin_memory=True, num_workers=4)
    # ======================================

    args.device = 'cpu'
    num_ftrs = 512
    if args.model == 'resnet18':
        num_ftrs = 512
    elif args.model == 'resnet34':
        num_ftrs = 1024
    elif args.model == 'resnet50':
        num_ftrs = 2048

    g_backbone = MyClassification(num_ftrs, len(classes))
    checkpoint_path = args.checkpoint
    if checkpoint_path != '':
        checkpoint = torch.load(checkpoint_path)
        g_backbone.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1
    idxs_users = [idx for idx in range(args.num_users)]
    idxs_users_sort = [[idx, len(dict_users[idx])] for idx in range(args.num_users)]
    idxs_users_sort = sorted(idxs_users_sort, key=lambda x:x[1], reverse=True)
    num_threads = args.thread

    tail_classed = 6
    global_tail_vec = torch.zeros(g_backbone.fc.weight[-tail_classed:].shape)
    grad_dim=global_tail_vec.shape[1]
    for rnd in tqdm(range(start_epoch, args.rounds+1)):
        backbone_w_locals, linear_w_locals, loss_locals = [0 for i in range(args.num_users)], [], []
        epoch_loss = []
        batch_loss = [0 for i in range(args.num_users)]
        print('\n===================round{}================='.format(rnd))

        for i in range(0, args.num_users, num_threads):
            processes = []
            torch.cuda.empty_cache()
            q = mp.Manager().Queue()
            for idx, _ in idxs_users_sort[i:i + num_threads]:
                p = mp.Process(target=local_train,
                               args=(q, copy.deepcopy(g_backbone), args, users_dataset_train[idx],
                                     idx, args.gpu))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            while not q.empty():
                fake_out = q.get()
                idx = int(fake_out[-1])
                backbone_w_locals[idx] = fake_out[0]
                batch_loss[idx] = fake_out[1]
            del q
        for idx in idxs_users:
            while isinstance(backbone_w_locals[idx], int):
                print('id:{} restart train.'.format(idx))
                q = mp.Manager().Queue()
                p = mp.Process(target=local_train, args=(q, copy.deepcopy(g_backbone), args, users_dataset_train[idx],
                                                         idx, args.gpu))
                p.start()
                p.join()
                fake_out = q.get()
                idx = int(fake_out[-1])
                backbone_w_locals[idx] = fake_out[0]
                batch_loss[idx] = fake_out[1]
                del q

        print('\nEpoch /{} \t loss={}'.format(args.rounds, sum(batch_loss)))

        dict_len = [len(dict_users[_idx]) for _idx in idxs_users]
        backbone_w_com_params = FedAvg_Rod(backbone_w_locals, dict_len, args.device)


        g_backbone.load_state_dict(backbone_w_com_params)

        # thread test
        torch.cuda.empty_cache()
        q_test = mp.Manager().Queue()
        p_test = mp.Process(target=globaltest15, args=(q_test, copy.deepcopy(g_backbone), test_loader, args.gpu))
        p_test.start()
        p_test.join()
        [average_auc, average_loss, average_acc, average_prec, average_rec, average_f1, average_ham, acc_per_class,
         global_3shot_acc] = q_test.get()

        del q_test

        print('round %d, average acc  %.3f, average prec  %.3f, average rec  %.3f, average f1  %.3f, average_ham  %.3f\n'
              % (rnd, average_acc, average_prec, average_rec,average_f1, average_ham))
        print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n' % (
            rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
        print('average_auc={},average_loss={}'.format(average_auc, average_loss))

        state = {
            'epoch': rnd + 1,
            'state': g_backbone.state_dict()
        }
        if not os.path.exists('./chest_save_model/model_15_pre_features/train_dataset{}/{}'.format(args.chest_data,args.model)):
            os.makedirs('./chest_save_model/model_15_pre_features/train_dataset{}/{}'.format(args.chest_data,args.model))
        checkpoint_path = os.path.join('./chest_save_model/model_15_pre_features/train_dataset{}/{}/checkpoint_train_{}_user_{}_lr_{}_epoch_'
                                       .format(args.chest_data, args.model, args.chest_data, args.num_users, args.lr) + str(rnd) + '.pth.tar')
        torch.save(state, checkpoint_path)