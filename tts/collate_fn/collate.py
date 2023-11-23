import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    raw_text_batch = []
    text_batch = []
    duration_batch = []
    mel_target_batch = []
    energy_batch = []
    pitch_batch = []
    mel_spec_path_batch = []
    alignment_path_batch = []
    max_spec_len = 0

    for item in dataset_items:
        raw_text_batch.append(item['raw_text'])
        text_batch.append(item['text'])
        duration_batch.append(item['duration'])
        max_spec_len = max(max_spec_len, item['mel_target'].size()[-1])
        mel_spec_path_batch.append(item['mel_spec_path'])
        alignment_path_batch.append(item['alignment_path'])
        energy_batch.append(item['energy'])
        pitch_batch.append(item['pitch'])
    
    for item in dataset_items:
        mel_target_batch.append(
            torch.nn.functional.pad(
                input=item['mel_target'], pad=(0, max_spec_len - item['mel_target'].size()[-1]),
                mode='constant', value=0
            )
        )

    mel_target_batch = torch.stack(mel_target_batch)
    text_batch = torch.nn.utils.rnn.pad_sequence(text_batch, batch_first=True, padding_value=0).int()
    duration_batch = torch.nn.utils.rnn.pad_sequence(duration_batch, batch_first=True, padding_value=0).int()
    pitch_batch = torch.nn.utils.rnn.pad_sequence(pitch_batch, batch_first=True, padding_value=0)
    energy_batch = torch.nn.utils.rnn.pad_sequence(energy_batch, batch_first=True, padding_value=0)

    return {
        "raw_text": raw_text_batch,
        "text": text_batch,
        "duration": duration_batch,
        "mel_target": mel_target_batch,
        "mel_spec_path": mel_spec_path_batch,
        "alignment_path": alignment_path_batch,
        "pitch": pitch_batch,
        "energy": energy_batch
    }
