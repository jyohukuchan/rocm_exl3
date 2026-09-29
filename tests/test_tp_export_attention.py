"""
TP export/import must round-trip every forward-affecting constructor flag of the attention modules.
Builds bare modules (no config, no children, no weights) so the test runs without a model; loading is
stubbed out because it needs a config and weights, which the kwargs plumbing under test does not.
"""
import sys, os, unittest
from unittest.mock import patch
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.attn import Attention
from exllamav3.modules.sliding_attn import SlidingAttention

HEAD_DIM, KV_HEADS, GQA = 64, 4, 2
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class FakeChild:
    """Stands in for a loaded Linear on both sides of the export/import."""
    def __init__(self, key = "fake"): self.key = key; self.device = DEVICE
    def tp_export(self, plan, producer): return {"cls": FakeChild}
    @staticmethod
    def tp_import_split(local_context, exported, plan, split): return FakeChild()


def _bare(cls, **flags):
    kw = dict(q_proj = FakeChild("q"), k_proj = FakeChild("k"), v_proj = FakeChild("v"), o_proj = FakeChild("o"))
    if cls is SlidingAttention: kw["sliding_window"] = 256
    m = cls(config = None, key = "model.layers.0.attn", layer_idx = 0, hidden_size = 512, head_dim = HEAD_DIM,
            num_q_heads = KV_HEADS * GQA, num_kv_heads = KV_HEADS, rope_settings = None, **kw, **flags)
    m.device = DEVICE
    return m


def _roundtrip(cls, first, last, **flags):
    m = _bare(cls, **flags)
    exported = m.tp_export(plan = {}, producer = None)
    plan = {m.key: (first, last, "heads")}
    with patch.object(cls, "load_local", lambda self, device, **kw: None), patch("torch.cuda.synchronize", lambda: None):
        imported = cls.tp_import({"device": DEVICE, "consumer": None}, exported, plan)
    return exported, imported


class TPExportAttentionTest(unittest.TestCase):

    def test_attention_flags_survive_export(self):
        for flags in ({"full_gate": True}, {"gate_softplus": True}, {"use_cu_seqlens": True}):
            exported, imported = _roundtrip(Attention, 0, 2, **flags)
            for k, v in flags.items():
                self.assertEqual(exported["kwargs"].get(k), v, f"Attention.tp_export drops {k}")
                self.assertEqual(getattr(imported, k), v, f"Attention.tp_import loses {k}")
            self.assertEqual(imported.num_kv_heads, 2)
            self.assertEqual(imported.num_q_heads, 2 * GQA)

    def test_sliding_attention_flags_survive_export(self):
        for flags in ({"full_gate": True}, {"gate_softplus": True}):
            exported, imported = _roundtrip(SlidingAttention, 1, 3, **flags)
            for k, v in flags.items():
                self.assertEqual(exported["kwargs"].get(k), v, f"SlidingAttention.tp_export drops {k}")
                self.assertEqual(getattr(imported, k), v, f"SlidingAttention.tp_import loses {k}")

    def test_attention_gate_split_follows_full_gate(self):
        # The g_proj split is applied through tp_import_split on the exported child; capture the split it receives
        seen = {}
        class FakeLinear:
            @staticmethod
            def tp_import_split(local_context, exported, plan, split):
                seen["split"] = split; return FakeChild()
        for full_gate, expect in ((False, (True, 2 * GQA, 4 * GQA)), (True, (True, 2 * GQA * HEAD_DIM, 4 * GQA * HEAD_DIM))):
            m = _bare(Attention, full_gate = full_gate)
            exported = m.tp_export(plan = {}, producer = None)
            exported["g_proj"] = {"cls": FakeLinear}
            with patch.object(Attention, "load_local", lambda self, device, **kw: None), patch("torch.cuda.synchronize", lambda: None):
                Attention.tp_import({"device": DEVICE, "consumer": None}, exported, {m.key: (2, 4, "heads")})
            self.assertEqual(seen["split"], expect, f"g_proj split wrong for full_gate={full_gate}")

    def test_attention_with_qsa_indexer_is_placed_whole(self):
        # QSA layers run whole on one rank: single-channel allocation with a module-enforced
        # device cap, the indexer travels with the export, and the owner rank gets it back
        class FakeIndexer(FakeChild):
            head_dim = 32; compress_ratio = 4
            def storage_size(self): return 0
            def tp_export(self, plan, producer): return {"cls": FakeIndexer, "key": self.key}
            @staticmethod
            def tp_import(local_context, exported, plan): return FakeIndexer(exported["key"])
        m = _bare(Attention)
        m.qsa_indexer = FakeIndexer("idx")
        exported = m.tp_export(plan = {}, producer = None)
        self.assertIsNotNone(exported.get("qsa_indexer"))
        plan = {m.key: (0, KV_HEADS, "heads")}
        with patch.object(Attention, "load_local", lambda self, device, **kw: None), patch("torch.cuda.synchronize", lambda: None):
            owner = Attention.tp_import({"device": DEVICE, "consumer": None}, exported, plan)
            stub = Attention.tp_import({"device": DEVICE, "consumer": None}, exported, {m.key: (KV_HEADS, KV_HEADS, "heads")})
        self.assertEqual(owner.qsa_indexer.key, "idx")
        self.assertIsNone(stub.qsa_indexer)
        with self.assertRaises(AssertionError):
            Attention.tp_import({"device": DEVICE, "consumer": None}, exported, {m.key: (0, KV_HEADS // 2, "heads")})

    def test_attention_qsa_allocation_is_single_device(self):
        # The allocator contract behind 'whole on one rank': one channel of width
        # num_kv_heads, a module-enforced max_devices = 1, and the indexer storage riding
        # along with the layer
        class FakeIndexer:
            def storage_size(self): return 4096
        m = _bare(Attention)
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(m, attr).storage_size = lambda: 0
            getattr(m, attr).recons_size = lambda: 0
        m.head_dim = HEAD_DIM
        m.qsa_indexer = FakeIndexer()
        tpa, = m.make_tp_allocation({})
        self.assertEqual(tpa.channels_to_split, 1)
        self.assertEqual(tpa.channel_width, KV_HEADS)
        self.assertEqual(tpa.max_devices, 1)
        self.assertGreaterEqual(tpa.storage_to_split, 4096)
        m.qsa_indexer = None
        tpa, = m.make_tp_allocation({})
        self.assertIsNone(tpa.max_devices)   # plain attention keeps splitting across ranks


if __name__ == "__main__":
    unittest.main()
