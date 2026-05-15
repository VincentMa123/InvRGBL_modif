import torch

from models.losses import region_consistency_loss_from_labels


def naive_region_loss(labels, values, valid_mask, min_region_pixels=3):
    losses = []
    for label in torch.unique(labels):
        if label.item() == 0:
            continue
        mask = (labels == label) & valid_mask
        if int(mask.sum().item()) < min_region_pixels:
            continue
        losses.append(values[mask].var(dim=0, unbiased=False).mean())
    if not losses:
        return values.sum() * 0.0
    return torch.stack(losses).mean()


def main():
    labels = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 0, 4, 5],
        ],
        dtype=torch.long,
    )
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    valid_mask[1, 3] = False

    values = (torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3) / 17.0)
    values.requires_grad_(True)

    loss = region_consistency_loss_from_labels(labels, values, valid_mask, min_region_pixels=3)
    expected = naive_region_loss(labels, values, valid_mask, min_region_pixels=3)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()

    ignored = (labels == 0) | (~valid_mask) | (labels == 2) | (labels == 5)
    assert values.grad[ignored].abs().sum().item() == 0.0
    assert values.grad[~ignored].abs().sum().item() > 0.0

    tiny_labels = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    tiny_values = torch.randn(2, 2, 3, requires_grad=True)
    tiny_loss = region_consistency_loss_from_labels(
        tiny_labels,
        tiny_values,
        torch.ones_like(tiny_labels, dtype=torch.bool),
        min_region_pixels=3,
    )
    assert tiny_loss.item() == 0.0
    tiny_loss.backward()
    assert tiny_values.grad is not None
    assert tiny_values.grad.abs().sum().item() == 0.0

    print("PASS: region_consistency_loss_from_labels matches naive loss and preserves gradients")


if __name__ == "__main__":
    main()
