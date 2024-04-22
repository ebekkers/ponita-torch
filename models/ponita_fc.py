from typing import Callable, Optional
import torch

from torch import nn
from torch import Tensor

from models.ponita import GridGenerator, PolynomialFeatures


class SeparableFiberBundleConvFC(nn.Module):
    __constants__ = ["depthwise_separable", "bias"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dim: int,
        bias: bool = True,
        depthwise_separable: Optional[bool] = False,
    ) -> None:
        super().__init__()

        if depthwise_separable and in_channels != out_channels:
            raise ValueError(
                "if depthwise_separable is True, should be in_channels == out_channels"
            )

        groups = out_channels if depthwise_separable else 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.depthwise_separable = depthwise_separable

        self.kernel = nn.Linear(kernel_dim, in_channels, bias=False)
        self.fiber_kernel = nn.Linear(
            kernel_dim, in_channels * out_channels // groups, bias=False
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer("bias", None)

        self.callibrate = True

    @torch.no_grad()
    def _callibrate(self, std_in, std_1, std_2):
        self.kernel.weight.data = self.kernel.weight.data * std_in / std_1
        self.fiber_kernel.weight.data = self.fiber_kernel.weight.data * std_1 / std_2
        self.callibrate = False

    def forward(
        self,
        x: Tensor,
        kernel_basis: Tensor,
        fiber_kernel_basis: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        kernel = self.kernel(kernel_basis)
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)

        if mask is None:
            x1 = torch.einsum("b n o c, b m n o c -> b m o c", x, kernel)
        else:
            x1 = torch.einsum("b n o c, b m n o c, b n -> b m o c", x, kernel, mask)

        if self.depthwise_separable:
            x2 = (
                torch.einsum("b m o c, p o c -> b m p c", x1, fiber_kernel)
                / self.out_channels
            )
        else:
            x2 = torch.einsum(
                "b m o c, p o d c -> b m p d",
                x1,
                fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)),
            ) / (self.in_channels * self.out_channels)

        if self.callibrate:
            _mask = ... if mask is None else mask.int()
            self._callibrate(*map(lambda x: x[_mask].std(), [x, x1, x2]))

        return x2 if self.bias is None else x2 + self.bias


class SeparableFiberBundleConvNextFC(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_dim: int,
        act_fn: Callable = nn.GELU(),
        layer_scale: Optional[float] = 1e-6,
        widening_factor: int = 4,
    ) -> None:
        super().__init__()

        self.conv = SeparableFiberBundleConvFC(
            channels, channels, kernel_dim, depthwise_separable=True
        )

        self.norm = nn.LayerNorm(channels)

        self.bottleneck = nn.Sequential(
            nn.Linear(channels, widening_factor * channels),
            act_fn,
            nn.Linear(widening_factor * channels, channels),
        )

        if layer_scale is not None:
            self.layer_scale = nn.Parameter(layer_scale * torch.ones(channels))
        else:
            self.register_buffer("layer_scale", None)

    def forward(
        self,
        x: Tensor,
        kernel_basis: Tensor,
        fiber_kernel_Basis: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x_res = x

        x = self.conv(x, kernel_basis, fiber_kernel_Basis, mask)
        x = self.norm(x)
        x = self.bottleneck(x)

        x = x if self.layer_scale is None else self.layer_scale * x

        x = x + x_res

        return x


class PonitaFC(nn.Module):
    __constants__ = ["global_pooling", "last_feature_conditioning"]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        output_dim_vec: int = 0,
        num_ori=20,
        basis_dim: Optional[int] = None,
        degree: int = 2,
        widening_factor: int = 4,
        layer_scale: Optional[float] = None,
        multiple_readouts: bool = True,
        last_feature_conditioning: bool = False,
        task_level: str = "graph",
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec

        self.num_ori = num_ori

        self.last_feature_conditioning = last_feature_conditioning

        self.register_buffer("ori_grid", GridGenerator(num_ori, steps=1000)())

        self.global_pooling = task_level == "graph"

        act_fn = nn.GELU()

        basis_dim = hidden_dim if basis_dim is None else basis_dim

        self.basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )
        self.fiber_basis_fn = nn.Sequential(
            PolynomialFeatures(degree),
            nn.LazyLinear(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, basis_dim),
            act_fn,
        )

        self.x_embedder = nn.Linear(input_dim, hidden_dim, bias=False)

        self.interaction_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()

        for i in range(num_layers):
            self.interaction_layers.append(
                SeparableFiberBundleConvNextFC(
                    hidden_dim,
                    basis_dim,
                    act_fn=act_fn,
                    layer_scale=layer_scale,
                    widening_factor=widening_factor,
                )
            )

            if multiple_readouts or i == (num_layers - 1):
                self.readout_layers.append(
                    nn.Linear(hidden_dim, output_dim + output_dim_vec)
                )
            else:
                self.readout_layers.append(None)

        self.register_buffer("_mask_default", torch.ones(1, 1))

    def forward(
        self, x: Tensor, pos: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        rel_pos = pos[:, None, :, None] - pos[..., None, None, :]

        invariant1 = (rel_pos * self.ori_grid[None, None, None]).sum(-1, keepdim=True)
        invariant2 = (rel_pos - rel_pos * invariant1).norm(dim=-1, keepdim=True)
        spatial_invariants = torch.cat((invariant1, invariant2), dim=-1)

        orientation_invariants = (self.ori_grid[:, None] * self.ori_grid[None]).sum(
            -1, keepdim=True
        )

        if self.last_feature_conditioning:
            noise_levels = x[..., 0, -1][:, None, None, None, None].expand(
                -1, *spatial_invariants.shape[1:-1], -1
            )
            spatial_invariants = torch.cat((spatial_invariants, noise_levels), dim=-1)

        kernel_basis = self.basis_fn(spatial_invariants)
        fiber_kernel_basis = self.fiber_basis_fn(orientation_invariants)

        # extend instead of repeat since they are copies
        x = self.x_embedder(x)[..., None, :].expand(-1, -1, self.num_ori, -1)

        readouts = 0
        num_readouts = 0

        for interaction_layer, readout_layer in zip(
            self.interaction_layers, self.readout_layers
        ):
            x = interaction_layer(x, kernel_basis, fiber_kernel_basis, mask)
            if readout_layer is not None:
                num_readouts += 1
                readouts = readouts + readout_layer(x)

        readouts = readouts / num_readouts

        readout_scaler, readout_vec = torch.split(
            readouts, [self.output_dim, self.output_dim_vec], dim=-1
        )

        output_scaler = readout_scaler.mean(dim=-2)
        output_vector = (
            torch.einsum("b n o c, o d -> b n c d", readout_vec, self.ori_grid)
            / self.num_ori
        )

        if self.global_pooling:
            mask = self._mask_default if mask is None else mask

            output_scaler = (output_scaler * mask[..., None]).sum(1) / mask.sum(1)[
                ..., None
            ]
            output_vector = (output_vector * mask[..., None, None]).sum(1) / mask.sum(
                1
            )[..., None, None]

        return output_scaler, output_vector


def main():
    device = "cuda"

    input = torch.randn(16, 24, 3, device=device)
    pos = torch.randn(16, 24, 3, device=device)

    mask = torch.round(torch.randn(16, 24, device=device))

    model = PonitaFC(3, 16, 7, 4, output_dim_vec=5, task_level="graph")
    model = model.to(device)

    scalar, vec = model(input, pos, mask=mask)

    loss = scalar.mean() + vec.mean()
    loss.backward()

    print(scalar.shape, vec.shape)


if __name__ == "__main__":
    main()
