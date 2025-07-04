import torch
import numpy as np


def mixup(feature: torch.Tensor, label: torch.Tensor,
          sequence_length: torch.Tensor,
          alpha: float):
    batch_size = feature.size(0)
    lam = np.random.beta(a=alpha, b=alpha,
                         size=(batch_size,)).astype(np.float32)
    lam = torch.from_numpy(lam).to(feature.device)
    lam_label = lam.unsqueeze(-1)
    lam_feature = lam_label.unsqueeze(-1)
    random_index = torch.randperm(batch_size)
    feature_shuf = feature[random_index]
    x = lam_feature * feature + (1 - lam_feature) * feature_shuf
    label_shuf = label[random_index]
    y = lam_label * label.float() + (1 - lam_label) * label_shuf.float()
    sequence_length_shuf = sequence_length[random_index]
    stacked_sequence_length = torch.stack(
        [sequence_length_shuf, sequence_length], dim=0)
    sequence_length = torch.max(stacked_sequence_length,
                                dim=0).values  # type: ignore
    return x, y, sequence_length, (lam, random_index)