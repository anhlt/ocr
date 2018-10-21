import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import base_model, np_to_tensor
import numpy as np
from torchvision.layers.roi_pool import ROIPool
from .utils.dataset import SOS_token, EOS_token
import random


class OCR(nn.Module):

    def __init__(self):
        pass


class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden_tensor):
        embedded = input_tensor.view(1, 1, -1)
        output = self.linear(embedded)
        output, hidden_tensor = self.gru(output, hidden_tensor)
        return output, hidden_tensor

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Decoder(nn.Module):
    """docstring for Dencoder"""

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        output = self.embedding(input_tensor).view(1, 1, -1)
        output = F.relu(output)

        output, hidden_tensor = self.gru(output, hidden_tensor)

        output = self.softmax(self.out(output[0]))
        return output, hidden_tensor

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Model(nn.Module):
    """docstring for Model"""

    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_hidden_size, decoder_output_size, lang=None, teacher_forcing_ratio=0.5, device="cuda:1"):
        super(Model, self).__init__()
        self.encoder_input_size = encoder_input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_output_size = decoder_output_size
        self.decoder_hidden_size = decoder_hidden_size
        self.device = torch.device('cuda')
        self.base_model = base_model().to(self.device)
        self.encoder = Encoder(self.encoder_input_size,
                               self.encoder_hidden_size).to(self.device)
        self.decoder = Decoder(self.decoder_hidden_size,
                               self.decoder_output_size).to(self.device)
        self.lang = lang

        self.roi_pool = ROIPool((7, 7), 1.0 / 16)
        self.use_teacher_forcing = True
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, input_tensor, target_tensor):
        assert(input_tensor.shape[0] == 1)
        loss = 0
        criterion = nn.NLLLoss()
        feature_map = self.base_model(input_tensor)
        feature_height, feature_width = feature_map.shape[2], feature_map.shape[3]
        all_anchors = self._create_anchors(
            feature_height, feature_width, feat_stride=16.)

        target_length = target_tensor.size(0)

        all_anchors = np_to_tensor(all_anchors, device=self.device)

        rois_features = self.roi_pool(feature_map, all_anchors)

        input_length = rois_features.shape[0]
        encoder_hidden = self.encoder.init_hidden().to(self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                rois_features[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        decoder_hidden = encoder_hidden
        if self.use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        return loss / target_length

    def _create_anchors(self, feature_height, feature_width, feat_stride=16.):
        shift_x = np.arange(0, feature_width, 2) * feat_stride
        shift_y = np.array([0] * shift_x.shape[0])

        shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).T
        # generate shifted anchors
        first_anchor = np.array(
            [0, 0, feature_height * feat_stride, feature_height * feat_stride])
        # move to specific gpu.
        # self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = first_anchor + shifts
        # add bbox deltas to shifted anchors to get proposal
        index_column = np.zeros((shift_x.shape[0], 1))
        all_anchors = np.hstack((index_column, all_anchors))

        return all_anchors

    def evaluate(self, input_tensor, max_length=300):
        with torch.no_grad():
            feature_map = self.base_model(input_tensor)
            feature_height, feature_width = feature_map.shape[2], feature_map.shape[3]
            all_anchors = self._create_anchors(
                feature_height, feature_width, feat_stride=16.)

            all_anchors = np_to_tensor(all_anchors, device=self.device)
            rois_features = self.roi_pool(feature_map, all_anchors)
            input_length = rois_features.shape[0]
            encoder_hidden = self.encoder.init_hidden().to(self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    rois_features[ei], encoder_hidden)

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.lang.index2char[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words
