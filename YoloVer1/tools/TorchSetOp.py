import torch


def bool_union_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return set_a | set_b


def bool_intersection_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return set_a & set_b


def bool_inverse_op(set_a: torch.Tensor):
    return ~set_a


def bool_difference_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return torch.where(bool_intersection_op(set_a, set_b), False, set_a)


def bool_gt_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return set_a > set_b


def bool_lt_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return set_a < set_b


def bool_eq_op(set_a: torch.Tensor, set_b: torch.Tensor):
    return set_a == set_b


def bool_mask_select(set_a: torch.Tensor, mask: torch.Tensor, default_value: torch.Tensor=None):
    if default_value is None:
        return set_a[mask]
    else:
        return torch.where(mask, set_a, default_value)
