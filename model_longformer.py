from __future__ import annotations

from torch import Tensor
from typing import cast
import torch
import torch.nn as nn
from jaxtyping import Int64
from transformers.models.longformer.modeling_longformer import (
    LongformerForSequenceClassification,
    LongformerSequenceClassifierOutput,
    LongformerConfig,
)


class Model(nn.Module):
    def __init__(self, use_pretrained=True):
        super(Model, self).__init__()

        self.labels = torch.tensor([0, 1], dtype=torch.long)
        cfg = LongformerConfig(
            num_labels=2,
            vocab_size=50265,
            max_position_embeddings=4098,
            type_vocab_size=1,
        )

        if use_pretrained:
            self.model = cast(
                LongformerForSequenceClassification,
                LongformerForSequenceClassification.from_pretrained(
                    "allenai/longformer-base-4096",
                    config=cfg,
                ),
            )
        else:
            self.model = LongformerForSequenceClassification(cfg)

    @staticmethod
    def from_ckpt(path: str) -> Model:
        statedict = torch.load(path, map_location="cpu")
        model = Model(use_pretrained=False)
        model.load_state_dict(statedict)
        
        return model

    def forward(
        self,
        x: Int64[Tensor, "B Seq"],
        target: Int64[Tensor, "B"],
        padding_mask: Int64[Tensor, "B Seq"],
    ) -> LongformerSequenceClassifierOutput:
        model_output = cast(
            LongformerSequenceClassifierOutput,
            self.model.forward(
                x, attention_mask=padding_mask, labels=target, return_dict=True
            ),
        )

        return model_output
