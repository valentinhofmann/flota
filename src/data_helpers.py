import numpy as np

import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, data):
        self.n_classes = len(set(data.label))
        self.texts = list(data.text)
        self.labels = list(data.label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


class ClassificationCollator:
    def __init__(self, tok, base, noise=0):
        self.tok = tok
        self.base = base
        self.noise = noise

    def __call__(self, batch):
        if self.noise:
            texts = [perturb(t, self.noise) for t, _ in batch]
        else:
            texts = [t for t, _ in batch]
        labels = torch.tensor([l for _, l in batch]).long()
        if not self.base:
            batch_tensors = self.tok(texts)
        else:
            batch_tensors = self.tok(texts, padding=True, truncation=True, return_tensors='pt')
        return batch_tensors, labels


def perturb(text, theta):
    text_split = text.strip().split()
    text_perturbed = [text_split[0]]
    for i in range(1, len(text_split)):
        rand_n = np.random.uniform(low=0.0, high=1.0)
        if rand_n <= theta:
            text_perturbed[-1] += text_split[i]
        else:
            text_perturbed.append(text_split[i])
    return ' '.join(text_perturbed)


def get_best(file):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                l_split = l.strip().split()
                results.append((float(l_split[0]), float(l_split[1]), float(l_split[2]), int(l_split[3])))
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None, None, None
