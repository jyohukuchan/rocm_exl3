"""HYV3 expert-width compatibility; no model weights or GPU are required."""

import json

import pytest

from exllamav3.architecture.hy_v3 import HyV3Config, HyV3Model
from exllamav3.architecture.hy_v3_mtp import HyV3MTPModel


@pytest.mark.parametrize("expert_config, expected", [
    ({"expert_hidden_dim": 768, "moe_intermediate_size": 1536}, 768),
    ({"expert_hidden_dim": 768}, 768),
    ({"moe_intermediate_size": 1536}, 1536),
])
def test_expert_width_propagates_to_trunk_and_mtp(tmp_path, expert_config, expected):
    config_dict = {
        "architectures": ["HYV3ForCausalLM"],
        "hidden_size": 2048,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 6912,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 2,
        "first_k_dense_replace": 1,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        **expert_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    config = HyV3Config(str(tmp_path))
    assert config.moe_intermediate_size == expected
    trunk = HyV3Model(config)
    draft = HyV3MTPModel(config)
    for model in (trunk, draft):
        modules = {module.key: module for module in model}
        sparse = [module for key, module in modules.items()
                  if key.endswith(".mlp") and hasattr(module, "num_experts")]
        assert len(sparse) == 1
        mlp = sparse[0]
        assert mlp.intermediate_size == expected
        assert mlp.gates[0].out_features == expected
        assert mlp.downs[0].in_features == expected
        assert mlp.shared_experts.gates[0].out_features == expected * 2
    config.stc.close()
