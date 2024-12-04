import torch

def generate_yolo_bboxes_test(scale_bboxes, head_size):
    assert len(scale_bboxes) == len(head_size), 'Number of scaled targets not match with detection heads'
        
    # Check if scale_bboxes contain proper tensor values
    for head_idx, s in enumerate(scale_bboxes):
        # Check tensor shape and type
        assert isinstance(s, torch.Tensor), f"Scale bbox {head_idx} is not a tensor"
        
        # Check value ranges
        assert torch.all((s[..., 0] >= 0) & (s[..., 0] <= 1)), f"Scale bbox {head_idx} has invalid objectness values"
        assert not torch.any(torch.isnan(s)), f"Scale bbox {head_idx} contains NaN values"
        assert not torch.any(torch.isinf(s)), f"Scale bbox {head_idx} contains Inf values"
    