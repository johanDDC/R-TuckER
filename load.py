from collections import defaultdict

from torch.utils.data import Dataset
from torch import tensor
import numpy as np

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
    def __init__(self, data, entity_vocab, relations_vocab, label_smoothing=None, test_set=False):
        self.data = data
        self.data_index = [(entity_vocab[self.data[i][0]], relations_vocab[self.data[i][1]],
                      entity_vocab[self.data[i][2]]) for i in range(len(self.data))]
        self.entity_relation_vocab = defaultdict(list) # dict of objects, which corresponds to given pair of subject and relation
        for triplet in self.data_index:
            self.entity_relation_vocab[(triplet[0], triplet[1])].append(triplet[2])
        self.entity_relation_pairs = list(self.entity_relation_vocab.keys()) # features for model
        self.size = (len(self.entity_relation_pairs), len(entity_vocab), len(relations_vocab))
        self.test_set = test_set
        self.label_smoothing = label_smoothing


    def __len__(self):
        return self.size[0]


    def _construct_targets(self, features):
        entity_id = features[0]
        relation_idx = features[1]
        targets = np.zeros((len(features), len(features), self.size[1]))
        for i, ent_id in enumerate(entity_id):
            for j, rel_id in enumerate(relation_idx):
                targets[i, j, self.entity_relation_vocab[(ent_id, rel_id)]] = 1
        return targets


    def __getitem__(self, idx):
        features = self.entity_relation_pairs[idx]
        targets = np.zeros(self.size[1])
        targets[self.entity_relation_vocab[features]] = 1
        if self.test_set:
            features = self.data_index[idx]
            return tensor(features), targets
        if self.label_smoothing:
            targets = (1 - self.label_smoothing) * targets + 1 / targets.shape[0]
        return tensor(features), targets

