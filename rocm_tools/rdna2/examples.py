"""The 8 fixed manifest examples for the single-V620 Phase 0-2 harness.

Rules: no network, no corpus downloads; every case is either an excerpt of a
repo-local evaluation text file (provenance = that path and the observed
content kind -- no author/licence claims beyond what the repo file itself
shows) or original fixed text embedded here (provenance = "authored for this
harness"). The set covers English prose, English list-style text, Japanese
prose and Python source. The combined tokenized length exceeds 1024 selectable
positions for the Qwen3 tokenizers with comfortable margin (~2x, estimated
without a tokenizer on the host; manifest.py hard-fails if it ever does not
fit), so `--positions 1024` always fits or says so.

The two repo-text cases read their source file at manifest creation time; if
the file is missing the manifest command fails rather than silently
substituting text, keeping the manifest digest stable for a given inputs tree
state.
"""

from __future__ import annotations

from pathlib import Path

# Repo-local eval corpus excerpts. Content kind below is what the files
# actually contain, observed 2026-09-28; case_ids mirror the source filenames.
_EN_TEXT_CASES = [
    {
        "case_id": "en_prose_pp_mod",
        "language": "en",
        "kind": "natural",
        "text_file": "eval/eval_texts/pride_prejudice_mod.txt",
        "provenance": (
            "first 800 chars of repo file eval/eval_texts/pride_prejudice_mod.txt "
            "(novel-style narrative prose per the repo filename/contents; authorship "
            "and licence beyond the repo file are not verified by this harness)"
        ),
    },
    {
        "case_id": "en_list_vm_char",
        "language": "en",
        "kind": "natural",
        "text_file": "eval/eval_texts/variable_man_char.txt",
        "provenance": (
            "first 800 chars of repo file eval/eval_texts/variable_man_char.txt "
            "(dramatis-personae style character/role list per the file contents; "
            "provenance beyond the repo path not verified by this harness)"
        ),
    },
]

# Original fixed examples, authored for this harness (2026-09). Embedded so
# manifest creation is reproducible from the repo alone.
_FIXED_CASES = [
    {
        "case_id": "en_tech_explain",
        "language": "en",
        "kind": "natural",
        "provenance": "authored for this harness (original expository prose)",
        "text": """\
A paged KV cache stores attention keys and values in fixed-size blocks instead of one
contiguous tensor per sequence. Paging makes the allocator simple: a sequence owns a
list of block numbers, blocks are handed out and reclaimed atomically, and a block table
maps logical positions to physical storage. The cost is that attention reads become
gathered loads rather than sequential ones, and the kernel must handle a ragged final
block. Prompt caching builds on the same blocks: the hash of a prefix's block contents
identifies reusable state, so a new request that shares a prefix skips recomputation of
those tokens. This only works when the hash covers every byte that a later read depends
on, and when evictions can never free a block another live sequence still points at.
On older GPUs the kernel and memory paths differ enough from newer parts that block
layouts and tile shapes tuned for recent silicon have to be revalidated on the actual
hardware rather than assumed.""",
    },
    {
        "case_id": "ja_natural_01",
        "language": "ja",
        "kind": "natural",
        "provenance": "authored for this harness (original Japanese prose)",
        "text": """\
量子化された言語モデルを古いGPUで動かすと、計算は合っているのに生成文が崩れることが
ある。原因の多くはカーネルの丸め誤差ではなく、同期やキャッシュ管理の境界条件にある。
特にページ単位でキーとバリューを保持する方式では、プロンプトの末尾がページ境界を
またぐときに古いデータを再読み込みすると、次のトークンの確率分布だけがおかしくなる。
検査には二つの方法がある。ひとつは完全に同一の入力列を教師ありで与え、各位置での
最尤トークンが基準と一致するかを数える方法だ。もうひとつは全語彙の分布距離、
つまりKLダイバージェンスを比較する方法で、こちらは下位トークンの差異にも反応する。
前者は成果物が小さく、後者は原因の切り分けに強い。現場ではまず前者で関門を設け、
失敗した位置だけを後者で詳しく調査するのが能率的である。""",
    },
    {
        "case_id": "ja_natural_02",
        "language": "ja",
        "kind": "natural",
        "provenance": "authored for this harness (original Japanese prose)",
        "text": """\
日本語テキストはトークン化の挙動が英語と大きく異なる。一文字あたりのトークン数は多く、
同じ漢字列でも前後の助詞の有無で区切りが変わることがある。そのため、量子化模型の
精度を比較するマニフェストには、必ずモデル自身のトークナイザで得た素のトークンID
列を保存しなければならない。別モデルのトークナイザで作り直したID列を渡すと、
同じ本文でも位置の対応が崩れ、比較が黙って間違ったものになる。対策として、マニフェスト
作成時の語彙サイズハッシュと入力ハッシュを計測結果へ写し込み、比較ツールはそれが
一致しない入力を弾く。これは実装の失敗ではなく、データの取り違えという実験の失敗を
早期に発見するための設計判断である。""",
    },
    {
        "case_id": "py_source_01",
        "language": "python",
        "kind": "code",
        "provenance": "authored for this harness (original Python source)",
        "text": '''\
import statistics
from dataclasses import dataclass, field


@dataclass
class RunStat:
    """Median/spread bookkeeping for a repeated timing measurement."""
    name: str
    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.samples.append(value)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def spread(self) -> float:
        med = self.median
        if not med:
            return 0.0
        return (max(self.samples) - min(self.samples)) / med

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n": len(self.samples),
            "median": round(self.median, 6),
            "spread": round(self.spread, 4),
            "flagged": self.spread > 0.05,
        }


def summarize(runs):
    rows = {}
    for run in runs:
        rows.setdefault(run.name, RunStat(run.name)).add(run.value)
    return [r.as_dict() for r in rows.values()]
''',
    },
    {
        "case_id": "py_source_02",
        "language": "python",
        "kind": "code",
        "provenance": "authored for this harness (original Python source)",
        "text": '''\
import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no incidental whitespace."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def digest(obj: Any, exclude: str = "_digest") -> str:
    body = {k: v for k, v in obj.items() if k != exclude}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def verify_integrity(obj: dict, exclude: str = "_digest") -> None:
    """Raise unless obj carries the canonical digest of its own content."""
    stored = obj.get(exclude)
    if not stored:
        raise ValueError(f"missing digest field {exclude!r}")
    recomputed = digest(obj, exclude=exclude)
    if stored != recomputed:
        raise ValueError(
            "digest mismatch: content was edited after signing "
            f"(stored {stored[:12]}... != recomputed {recomputed[:12]}...)"
        )


def merge_manifests(a: dict, b: dict) -> dict:
    """Refuse to merge manifests that disagree on any literal input."""
    for case in b["cases"]:
        cid = case["case_id"]
        other = next((c for c in a["cases"] if c["case_id"] == cid), None)
        if other is not None and other["ids"] != case["ids"]:
            raise ValueError(f"case {cid} ids differ between manifests")
    return {"cases": a["cases"] + [
        c for c in b["cases"]
        if not any(o["case_id"] == c["case_id"] for o in a["cases"])
    ]}
''',
    },
    {
        "case_id": "py_source_03",
        "language": "python",
        "kind": "code",
        "provenance": "authored for this harness (original Python source)",
        "text": '''\
import math
from collections import Counter


def allocate_positions(lengths, total):
    """Split `total` probe positions over cases, proportional to case length.

    Uses the largest-remainder method so the result is deterministic and each
    case keeps at least one position whenever that is possible at all.
    """
    n = len(lengths)
    if total < n:
        raise ValueError("need at least one position per case")
    if total > sum(lengths):
        raise ValueError("more positions requested than available")
    s = sum(lengths)
    raw = [total * L / s for L in lengths]
    counts = [max(1, min(L, math.floor(r))) for r, L in zip(raw, lengths)]
    remainder = total - sum(counts)
    order = sorted(
        range(n), key=lambda i: (-(raw[i] - math.floor(raw[i])), i)
    )
    while remainder > 0:
        progressed = False
        for i in order:
            if counts[i] < lengths[i]:
                counts[i] += 1
                remainder -= 1
                progressed = True
                if not remainder:
                    break
        if not progressed:
            raise AssertionError("stalled")
    return counts


def agreement(ref_top1, cand_top1):
    if len(ref_top1) != len(cand_top1):
        raise ValueError("length mismatch")
    hits = sum(1 for a, b in zip(ref_top1, cand_top1) if a == b)
    rate = hits / len(ref_top1) if ref_top1 else 0.0
    return Counter(match=hits, total=len(ref_top1)), round(rate, 6)
''',
    },
]


def get_cases(repo_root: Path) -> list[dict]:
    """
    The fixed 8-case example set. Each entry: case_id, language, kind, text,
    provenance (and the source path for file-backed cases). File-backed text is
    truncated deterministically (first 800 chars) so the case lengths don't
    drift with unrelated edits to the eval corpus files.
    """
    cases: list[dict] = []
    for spec in _EN_TEXT_CASES:
        path = repo_root / spec["text_file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"manifest example source missing: {path} "
                f"(case {spec['case_id']}); repo eval_texts must exist for "
                f"deterministic manifest creation")
        text = path.read_text(encoding = "utf-8").strip()
        text = text[:800]  # deterministic truncation; full file remains hashed in provenance
        cases.append({
            "case_id": spec["case_id"],
            "language": spec["language"],
            "kind": spec["kind"],
            "text": text,
            "provenance": spec["provenance"],
            "source_file": str(path.relative_to(repo_root)),
            "source_file_sha256": hashlib_sha256_file(path),
        })
    for spec in _FIXED_CASES:
        cases.append({
            "case_id": spec["case_id"],
            "language": spec["language"],
            "kind": spec["kind"],
            "text": spec["text"],
            "provenance": spec["provenance"],
        })
    return cases


def hashlib_sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
