from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch import tensor

class Data:
    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities


class KG_dataset(Dataset):
    def __init__(self, data, dataset, label_smoothing=None, test_set=False):
        self.data = dataset
        self.entity_index = {data.entities[i]: i for i in range(len(data.entities))}
        self.relation_index = {data.relations[i]: i for i in range(len(data.relations))}
        self.data_index = self.__get_data_idx(self.data)
        self.entity_relation_vocab = self.__get_er_vocab(self.data_index) # dict of objects, which corresponds to given pair of subject and relation
        self.entity_relation_pairs = list(self.entity_relation_vocab.keys()) # features for model
        if test_set:
            full_data_index = self.__get_data_idx(data.data)
            self.entity_relation_vocab = self.__get_er_vocab(full_data_index)

        self.size = (len(self.entity_relation_pairs), len(self.entity_index), len(self.relation_index))
        self.test_set = test_set
        self.label_smoothing = 0 if test_set else label_smoothing


    def __get_data_idx(self, data):
        return [(self.entity_index[data[i][0]], self.relation_index[data[i][1]],
                  self.entity_index[data[i][2]]) for i in range(len(data))]

    def __get_er_vocab(self, data_idx):
        data = data_idx
        er_vocab = defaultdict(list)
        for triplet in data:
            er_vocab[(triplet[0], triplet[1])].append(triplet[2])
        return er_vocab

    def __len__(self):
        if self.test_set:
            return len(self.data_index)
        else:
            return self.size[0]

    def __getitem__(self, idx):
        targets = torch.zeros(self.size[1])
        if self.test_set:
            features = self.data_index[idx]
            feature_pair = (features[0], features[1])
            targets[self.entity_relation_vocab[feature_pair]] = 1
        else:
            features = self.entity_relation_pairs[idx]
            targets[self.entity_relation_vocab[features]] = 1
        if self.label_smoothing:
            targets = (1 - self.label_smoothing) * targets + 1 / self.size[1]
        return tensor(features), targets.float()
