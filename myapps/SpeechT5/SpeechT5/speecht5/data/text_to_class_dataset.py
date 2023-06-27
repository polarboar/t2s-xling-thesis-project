# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import itertools
import logging
import os
from typing import Any, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
import librosa
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset


logger = logging.getLogger(__name__)

def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
):
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets

def load_text_and_classes(text_path, class_path):
    classes = []
    text = []
    sizes = []
    with open(text_path) as f:
        for line in f:
            data = line.strip()
            text.append(data)
            sizes.append(len(data))

    with open(class_path) as f:
        for line in f:
            data = line.strip()
            classes.append(data)

    return text, classes, sizes

class TextToClassDataset(FairseqDataset):
    def __init__(
        self,
        text_path: str,
        class_path: str,
        text_processors: Optional[List[Any]] = None,
        class_processors: Optional[List[Any]] = None,
        shuffle: bool = True,
        store_labels: bool = True,
        src_dict: Optional[Dictionary] = None,
        tgt_dict: Optional[Dictionary] = None,
        tokenizer = None,
        reduction_factor: int = 1,
    ):
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        #self.num_labels = len(label_paths)

        self.text_processors = text_processors
        self.class_processors = class_processors

        self.text, self.classes, self.text_sizes = load_text_and_classes(text_path, class_path)

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.store_labels = store_labels

        self.reduction_factor = reduction_factor

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]
    
    def get_label(self, index):
        label = self.text_classes[index]
        if self.class_processors is not None:
            label = self.class_processors(label)

        return label

    def get_class(self, index):
        label = self.classes[index]
        if self.class_processors is not None:
            label = self.class_processors(label)

        return label

    def get_classes(self, index):
        return [self.get_class(index, i) for i in range(self.num_labels)]
    
    def get_source(self, index):
        source = self.text[index]
        if self.tokenizer is not None:
            source = self.tokenizer.encode(source)

        if self.text_processors is not None:
            source = self.text_processors(source)

        return source

    def __getitem__(self, index):
        label = self.get_class(index)
        source = self.get_source(index)
        return {"id": index, "source": source, "label": label}
    
    def __len__(self):
        return len(self.text)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        texts = [s["source"] for s in samples]
        text_sizes = [len(s) for s in texts]

        text_size = max(text_sizes)
        #collated_texts, padding_mask = self.collater_text(
        #    texts, text_size
        #) 
        sources_list, sources_lengths_list, sources_ntokens_list = self.collater_text([texts]) 

        decoder_label = None
        decoer_taget = None
        decoder_target_lengths = None
        if samples[0]["label"] is not None:
            targets_by_label = [
                [s["label"] for s in samples]
            ]
            targets_list, lengths_list, ntokens_list = self.collater_label(targets_by_label)

            decoder_label = [
                (targets_list[0][i, :lengths_list[0][i]]).long()
                for i in range(targets_list[0].size(0))
            ]

            decoder_target = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            decoder_target_lengths = torch.tensor(
                [x.size(0) for x in decoder_label], dtype=torch.long
            )
        prev_output_tokens = data_utils.collate_tokens(
            [torch.LongTensor([-1]) for _ in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )

        net_input = {
            "src_tokens": sources_list[0], 
            "src_lengths": sources_lengths_list[0],
            "prev_output_tokens": prev_output_tokens, 
            "task_name": "t2c",
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "target": decoder_target,
            "target_lengths": decoder_target_lengths,
            "task_name": "t2c",
            "ntokens": len(samples),
        }

        return batch
    
    """
    def collater_text(self, texts, text_size):
        collated_texts = texts[0].new_zeros(len(texts), text_size)
        padding_mask = (
            torch.BoolTensor(collated_texts.shape).fill_(False)
        )
        for i, text in enumerate(texts):
            diff = len(text) - text_size
            if diff == 0:
                collated_texts[i] = text
            elif diff < 0:
                collated_texts[i] = torch.cat([text, text.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return collated_texts, padding_mask
    """

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens
    
    def collater_text(self, sources_by_label):
        sources_list, lengths_list, ntokens_list = [], [], []
        itr = zip(sources_by_label, [self.src_dict.pad()])
        for sources, pad in itr:
            sources, lengths, ntokens = self.collater_seq_label(sources, pad)
            sources_list.append(sources)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return sources_list, lengths_list, ntokens_list

    def collater_label(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, [self.tgt_dict.pad()])
        for targets, pad in itr:
            targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.text_sizes[index]

    @property
    def sizes(self):
        return np.array(self.text_sizes)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.text_sizes)
        return np.lexsort(order)[::-1]

