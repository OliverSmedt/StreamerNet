import torch
import numpy as np
def accuracy(logits, labels):
    pred_probability = torch.nn.Softmax(dim=1)(logits)
    predicted_category = pred_probability.argmax(1)[0]
    labels = labels.numpy()
    ratio = np.sum(np.where(labels == predicted_category, 1, 0))/len(labels)
    return ratio
