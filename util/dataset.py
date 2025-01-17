from http import client
from imghdr import tests
from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
# from yaml import DirectiveToken
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils
from util.imbalance_cifar import IMBALANCECIFAR10,IMBALANCECIFAR100
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class myDataset():
    def __init__(self, args):
        self.m_args = args
        
    def get_args(self):
        return self.m_args

    def get_imbalanced_dataset(self, args):
        if args.dataset == 'cifar10':
            data_path = './cifar_lt/'
            args.num_classes = 10
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            dataset_train = IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
            # data_test_tail =
            # dataset_localtest= IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            # print(len(dataset_localtest))
            n_test = len(dataset_test)
            y_test = np.array(dataset_test.targets)
        else:
            exit('Error: unrecognized dataset')

        if args.iid:
            print("Into iid sampling")
            dict_users = iid_sampling(n_train, args.num_users, args.seed)
        
        else:
            print("Into non-iid sampling")
            dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        clients_sizes= [len(dict_users[i]) for i in range(args.num_users)]
        print("clients_sizes:{}".format(clients_sizes))
        # print(len(dataset_test))

        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(10000):
                if y_test[j]==i:
                    idxs.append(j)
            assert 1000==len(idxs)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes


        alist = np.array([[np.sum(y_train[list(dict_users[i])]==j) for j in range(10)] for i in range(len(clients_sizes))])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        distributions = np.array([[alist[i][j]/sum(alist.sum(0)) for j in range(10)] for i in range(len(clients_sizes)) ])

        testsizes = np.array([[int(distributions[i][j]*10000) for j in range(10)] for i in range(len(clients_sizes)) ])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")  # close to 10000
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))
        # pdb.set_trace()
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = min(testsizes[i][j], 1000)
                temp=np.random.choice(map_testset[j], cnt, replace=False)
                for m in range(len(temp)):
                    idxs.append(temp[m])
            dict_localtest[i] = set(idxs)
        # print(dict_localtest)
        blist = np.array([[np.sum(y_test[list(dict_localtest[i])]==j) for j in range(10)] for i in range(len(clients_sizes))])
        # frac = sum(testsizes.sum(1))
        assert testsizes.all()==blist.all()
        assert len(dict_users) == len(dict_localtest) ==args.num_users

        # add to member variable
        # self.clients_sizes = clients_sizes      # clients_sizes[i]: sample numbers of client i
        self.training_set_distribution = alist  # training_set_distribution[i, j]: sample size of clien i and class j
        self.local_test_distribution = testsizes    # local_test_distribution[i, j]: sample size of clien i and class j
        self.global_test_distribution = np.sum(testsizes, axis=0)
        return dataset_train, dataset_test, dict_users, dict_localtest,alist

    def get_imbalanced_dataset100(self, args):
        data_path = './cifar_lt/'
        args.num_classes = 100
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = IMBALANCECIFAR100(data_path, imb_factor=args.IF, train=True, download=True,
                                              transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

        y_test = np.array(dataset_test.targets)

        if args.iid:
            print("Into iid sampling")
            dict_users = iid_sampling(n_train, args.num_users, args.seed)

        else:
            print("Into non-iid sampling")
            dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users,
                                                    args.seed, args.alpha_dirichlet)
        clients_sizes = [len(dict_users[i]) for i in range(args.num_users)]
        print("clients_sizes:{}".format(clients_sizes))
        # print(len(dataset_test))

        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(10000):
                if y_test[j] == i:
                    idxs.append(j)
            assert 100 == len(idxs)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes

        alist = np.array(
            [[np.sum(y_train[list(dict_users[i])] == j) for j in range(100)] for i in range(len(clients_sizes))])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        distributions = np.array(
            [[alist[i][j] / sum(alist.sum(0)) for j in range(100)] for i in range(len(clients_sizes))])

        testsizes = np.array(
            [[int(distributions[i][j] * 10000) for j in range(100)] for i in range(len(clients_sizes))])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")  # close to 10000
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))
        # pdb.set_trace()
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = min(testsizes[i][j], len(map_testset[j]))
                temp = np.random.choice(map_testset[j], cnt, replace=False)
                for m in range(len(temp)):
                    idxs.append(temp[m])
            dict_localtest[i] = set(idxs)
        # print(dict_localtest)
        blist = np.array(
            [[np.sum(y_test[list(dict_localtest[i])] == j) for j in range(100)] for i in range(len(clients_sizes))])
        # frac = sum(testsizes.sum(1))
        assert testsizes.all() == blist.all()
        assert len(dict_users) == len(dict_localtest) == args.num_users
        self.training_set_distribution = alist
        self.local_test_distribution = testsizes
        self.global_test_distribution = np.sum(testsizes, axis=0)
        return dataset_train, dataset_test, dict_users, dict_localtest, alist

    def get_imbalanced_dataset_imagenet(self, args):
        # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu !=-1 else 'cpu')
        args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        if args.dataset == 'imagenet':
            data_path = './imagenet/'
            args.num_classes = 1000
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            dataset_train = IMBALANCECIFAR100(data_path, imb_factor=args.IF, train=True, download=True,
                                              transform=trans_train)
            dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
            # data_test_tail =

            # dataset_localtest= IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            # print(len(dataset_localtest))
            n_test = len(dataset_test)
            y_test = np.array(dataset_test.targets)
        else:
            exit('Error: unrecognized dataset')

        if args.iid:
            print("Into iid sampling")
            dict_users = iid_sampling(n_train, args.num_users, args.seed)

        else:
            print("Into non-iid sampling")
            dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users,
                                                    args.seed, args.alpha_dirichlet)
        clients_sizes = [len(dict_users[i]) for i in range(args.num_users)]
        print("clients_sizes:{}".format(clients_sizes))
        # print(len(dataset_test))

        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(10000):
                if y_test[j] == i:
                    idxs.append(j)
            assert 100 == len(idxs)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes

        alist = np.array(
            [[np.sum(y_train[list(dict_users[i])] == j) for j in range(100)] for i in range(len(clients_sizes))])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        distributions = np.array(
            [[alist[i][j] / sum(alist.sum(0)) for j in range(100)] for i in range(len(clients_sizes))])

        testsizes = np.array(
            [[int(distributions[i][j] * 10000) for j in range(100)] for i in range(len(clients_sizes))])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")  # close to 10000
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))
        # pdb.set_trace()
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = min(testsizes[i][j], len(map_testset[j]))
                temp = np.random.choice(map_testset[j], cnt, replace=False)
                for m in range(len(temp)):
                    idxs.append(temp[m])
            dict_localtest[i] = set(idxs)
        # print(dict_localtest)
        blist = np.array(
            [[np.sum(y_test[list(dict_localtest[i])] == j) for j in range(100)] for i in range(len(clients_sizes))])
        # frac = sum(testsizes.sum(1))
        assert testsizes.all() == blist.all()
        assert len(dict_users) == len(dict_localtest) == args.num_users
        # print(dict_localtest)
        # print(dict_users[i])
        # pdb.set_trace()
        # print(frac)

        # add to member variable
        # self.clients_sizes = clients_sizes      # clients_sizes[i]: sample numbers of client i
        self.training_set_distribution = alist  # training_set_distribution[i, j]: sample size of clien i and class j
        self.local_test_distribution = testsizes  # local_test_distribution[i, j]: sample size of clien i and class j
        self.global_test_distribution = np.sum(testsizes, axis=0)
        return dataset_train, dataset_test, dict_users, dict_localtest, alist

    def get_balanced_dataset(self, args):
        if args.dataset == 'cifar10':
            data_path = './cifar_lt/'
            args.num_classes = 10
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            y_test = np.array(dataset_test.targets)
        else:
            exit('Error: unrecognized dataset')


        dict_users = iid_sampling(n_train, args.num_users, args.seed)
        
        for i in range(args.num_users):
            # create a longtailed distribution in client i
            cls_num=10
            client_size = len(dict_users[i])
            # print(client_size)
            img_max = client_size/ cls_num
            lt_sizes = []
            for cls_idx in range(cls_num):
                num = img_max * (args.IF**(cls_idx / (cls_num - 1.0)))
                lt_sizes.append(int(num))
            
            head = i%cls_num #decide the head class in longtailed distribution
            for j in range(10): # for each class
                cur_cls = (head+j)%10   #current class idx
                target_cls_size = lt_sizes[j]
                labellist = y_train[list(dict_users[i])]==cur_cls
                cur_cls_size = np.sum(labellist)
                indices= []
                for (idx,v) in enumerate(labellist):
                    if v==True:
                        indices.append(list(dict_users[i])[idx])
            
                assert len(indices)==cur_cls_size
                for n in range(len(indices)):
                    assert y_train[indices[n]]==cur_cls
                if target_cls_size== cur_cls_size or target_cls_size>cur_cls_size:
                    print('the current class doesnt need dropout')
                    continue
                elif target_cls_size<cur_cls_size:
                    clientlist = list(dict_users[i])
                    cnt = cur_cls_size-target_cls_size
                    for m in range(cnt):
                        clientlist.remove(indices[m])
                        # pop one instance of class j from dict_user[i]
                    dict_users[i] = set(clientlist)
            
        alist = np.array([[np.sum(y_train[list(dict_users[i])]==j) for j in range(10)] for i in range(40)])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        distributions = np.array([[alist[i][j]/sum(alist.sum(0)) for j in range(10)] for i in range(args.num_users) ])

        testsizes = np.array([[int(distributions[i][j]*10000) for j in range(10)] for i in range(args.num_users) ])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")  # close to 10000
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))

        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(10000):
                if y_test[j]==i:
                    idxs.append(j)
            assert 1000==len(idxs)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = testsizes[i][j]
                temp=np.random.choice(map_testset[j], cnt, replace=False)
                for m in range(len(temp)):
                    idxs.append(temp[m])
            dict_localtest[i] = set(idxs)
        # print(dict_localtest)
        blist = np.array([[np.sum(y_test[list(dict_localtest[i])]==j) for j in range(10)] for i in range(args.num_users)])
        # frac = sum(testsizes.sum(1))
        assert testsizes.all()==blist.all()
        assert len(dict_users) == len(dict_localtest) ==args.num_users
        # pdb.set_trace()
        return dataset_train, dataset_test, dict_users, dict_localtest,alist

    def get_balanced_dataset100(self, args):
        args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        if args.dataset == 'cifar100':
            data_path = './cifar_lt/'
            args.num_classes = 100
            args.num_users=80
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
            )
            dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            y_test = np.array(dataset_test.targets)
        else:
            exit('Error: unrecognized dataset')

        dict_users = iid_sampling(n_train, args.num_users, args.seed)

        for i in range(args.num_users):
            # create a longtailed distribution in client i
            cls_num = 100
            client_size = len(dict_users[i])
            # print(client_size)
            img_max = client_size / cls_num
            lt_sizes = []
            for cls_idx in range(cls_num):
                num = img_max * (args.IF ** (cls_idx / (cls_num - 1.0)))
                lt_sizes.append(int(num))

            head = i % cls_num  # decide the head class in longtailed distribution
            for j in range(100):  # for each class
                cur_cls = (head + j) % 100  # current class idx
                target_cls_size = lt_sizes[j]
                labellist = y_train[list(dict_users[i])] == cur_cls
                cur_cls_size = np.sum(labellist)
                indices = []
                for (idx, v) in enumerate(labellist):
                    if v == True:
                        indices.append(list(dict_users[i])[idx])

                assert len(indices) == cur_cls_size
                for n in range(len(indices)):
                    assert y_train[indices[n]] == cur_cls
                if target_cls_size == cur_cls_size or target_cls_size > cur_cls_size:
                    print('the current class doesnt need dropout')
                    continue
                elif target_cls_size < cur_cls_size:
                    clientlist = list(dict_users[i])
                    cnt = cur_cls_size - target_cls_size
                    for m in range(cnt):
                        clientlist.remove(indices[m])
                        # pop one instance of class j from dict_user[i]
                    dict_users[i] = set(clientlist)

        alist = np.array([[np.sum(y_train[list(dict_users[i])] == j) for j in range(100)] for i in range(40)])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        distributions = np.array([[alist[i][j] / sum(alist.sum(0)) for j in range(100)] for i in range(args.num_users)])

        testsizes = np.array([[int(distributions[i][j] * 10000) for j in range(100)] for i in range(args.num_users)])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")  # close to 10000
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))

        map_testset = {}
        for i in range(args.num_classes):
            idxs = []
            for j in range(10000):
                if y_test[j] == i:
                    idxs.append(j)
            assert 1000 == len(idxs)
            map_testset[i] = idxs
        assert len(map_testset) == args.num_classes
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = testsizes[i][j]
                temp = np.random.choice(map_testset[j], cnt, replace=False)
                for m in range(len(temp)):
                    idxs.append(temp[m])
            dict_localtest[i] = set(idxs)
        # print(dict_localtest)
        blist = np.array(
            [[np.sum(y_test[list(dict_localtest[i])] == j) for j in range(100)] for i in range(args.num_users)])
        # frac = sum(testsizes.sum(1))
        assert testsizes.all() == blist.all()
        assert len(dict_users) == len(dict_localtest) == args.num_users
        # pdb.set_trace()
        return dataset_train, dataset_test, dict_users, dict_localtest, alist


    def get_tail_dataset(self, dataset_test, tails=[]):
        def filter_func(item):
            _, label = item
            return label in tails
        return list(filter(filter_func, dataset_test))

    def train_test(self, dataset, idxs):

        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.m_args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test


