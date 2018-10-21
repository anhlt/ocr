from torchvision.models.squeezenet import SqueezeNet, squeezenet1_1
import torch.nn as nn
import torch


def base_model():
    full_model = squeezenet1_1(pretrained=True)
    feature_model = nn.Sequential(*list(full_model.features.children())[:-1])
    return feature_model


def np_to_tensor(x, device=torch.device('cuda'), dtype=torch.FloatTensor):
    v = torch.from_numpy(x).type(dtype).to(device)
    return v
