import torch
import sys
sys.path.insert(0, "src")
from emotion.fusion import CrossAttentionFusion


def test_fusion_shape():
    fusion = CrossAttentionFusion(dim=256, num_heads=4)
    v = torch.randn(2, 15, 256)
    a = torch.randn(2, 15, 256)
    out = fusion(v, a)
    assert out.shape == (2, 256)


def test_fusion_deterministic():
    fusion = CrossAttentionFusion()
    fusion.eval()
    v = torch.ones(1, 15, 256) * 0.5
    a = torch.ones(1, 15, 256) * 0.5
    with torch.no_grad():
        o1 = fusion(v, a)
        o2 = fusion(v, a)
    assert torch.allclose(o1, o2)
