import torch
import torch.nn as nn
from transformers import PreTrainedModel


class LMKTWithKCEmbedding(nn.Module):
    """Wrapper that adds a trainable logit_scale for ordinal IRT prediction."""

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.logit_scale = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, position_ids):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False
        )

    def save_pretrained(self, path):
        self.base_model.save_pretrained(path)
        torch.save({'logit_scale': self.logit_scale.data}, f"{path}/irt_heads.pt")

    def load_mlp_head(self, path):
        checkpoint = torch.load(f"{path}/irt_heads.pt")
        self.logit_scale.data = checkpoint['logit_scale']

    def print_trainable_parameters(self):
        lora_trainable_params = 0
        lora_all_params = 0
        if hasattr(self.base_model, 'print_trainable_parameters'):
            for name, param in self.base_model.named_parameters():
                lora_all_params += param.numel()
                if param.requires_grad:
                    lora_trainable_params += param.numel()
            print("=" * 60)
            print("Base model (LoRA) trainable parameters:")
            self.base_model.print_trainable_parameters()

        print("=" * 60)
        print(f"Ordinal IRT logit_scale: 1 trainable parameter")

        total_trainable = lora_trainable_params + 1
        total_all = lora_all_params + 1
        trainable_percent = 100 * total_trainable / total_all if total_all > 0 else 0
        print("=" * 60)
        print(f"TOTAL TRAINABLE PARAMETERS: {total_trainable:,}")
        print(f"  - LoRA parameters: {lora_trainable_params:,}")
        print(f"  - logit_scale: 1")
        print(f"  - Percentage of all parameters: {trainable_percent:.4f}%")
        print("=" * 60)