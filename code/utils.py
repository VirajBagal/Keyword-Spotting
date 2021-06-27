import torch
import numpy as np
from torchaudio.transforms import Resample, MelSpectrogram, MFCC
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

def mixup(imgs, labels, alpha):
  lam = np.random.beta(alpha,alpha)
  index = torch.randperm(len(imgs))
  shuffled_imgs = imgs[index]
  shuffled_labels = labels[index]
  new_imgs = lam*imgs + (1-lam)*shuffled_imgs

  return new_imgs, shuffled_labels, lam 

def label_to_index(labels, word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(labels, index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(args, batch):
    # Make all tensor in a batch the same length by padding with zeros

    if args.input_type == 'waveform':

        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)

        return batch.permute(0, 2, 1)
    else:
        batch = [item.permute(2, 0, 1) for item in batch]  # make the sequence length as the first dimension
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        batch = batch.repeat(1, 1, 3, 1)   # make it 3 channel
        return batch.permute(0, 2, 3, 1)



def collate_fn(batch, args, labels_list, transform, feature_extractor = None):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    if feature_extractor:
        resampled_list, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            resampled = transform(waveform).squeeze(0).numpy()
            resampled_list.append(resampled)
            targets += [label_to_index(labels_list, label)]

        feat = feature_extractor(resampled_list, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask = True)

        # Group the list of tensors into a batched tensor
        # tensors = pad_sequence(tensors)
        inputids = feat['input_values']
        targets = torch.stack(targets)

    else:
        resampled_list, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            if args.input_type != 'waveform':
                resampled = transform(waveform).squeeze(1)
            else:
                resampled = transform(waveform).numpy()

            resampled_list.append(resampled)
            targets += [label_to_index(labels_list, label)]

        # Group the list of tensors into a batched tensor
        inputids = pad_sequence(args, resampled_list)
        targets = torch.stack(targets)


    return inputids, targets




@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
