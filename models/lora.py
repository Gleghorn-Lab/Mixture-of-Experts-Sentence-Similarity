import torch
import torch.nn as nn


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        """
        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            bias (bool): Whether to use a bias term.
            r (int): Rank of the low-rank update matrices.
            lora_alpha (float): Scaling factor for the low-rank update.
            lora_dropout (float): Dropout probability applied to the input before the LoRA update.
        """
        # Initialize as a standard Linear layer.
        super().__init__(in_features, out_features, bias)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        if r > 0:
            # lora_A: shape (r, in_features), initialized with small random values.
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            # lora_B: shape (out_features, r), initialized to zeros so the initial update is zero.
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None

        # Freeze the pretrained weight and bias so that only LoRA parameters are updated.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the standard linear transformation using the frozen weights.
        result = super().forward(x)
        if self.r > 0:
            # Compute the low-rank update.
            # x shape: (..., in_features)
            # lora_A.t() shape: (in_features, r)
            lora_update = self.lora_dropout(x) @ self.lora_A.t()  # shape: (..., r)
            # lora_B.t() shape: (r, out_features)
            lora_update = lora_update @ self.lora_B.t()  # shape: (..., out_features)
            result = result + self.scaling * lora_update
        return result
