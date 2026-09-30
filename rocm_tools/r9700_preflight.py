#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r9700_preflight.py - R9700 (gfx1201) 環境の読み取り専用プリフライト診断。

EXL3 をビルド/ロードする前に、現在の環境が R9700 と比較ワークアラウンド
前提の運用に適合しているかを検査する。副作用はない:

- torch はハードウェア照会時のみ遅延 import する (torch.cuda のデバイス
  プロパティのみ参照し、モデルやネイティブ拡張は一切ロードしない)。
- exllamav3 の import、JIT ビルド、サブプロセス起動、ファイル書き込み、
  環境変更は一切行わない。
- 環境変数はビルドターゲット / fused-MoE 関連の 3 変数しか記録しない
  (無関係な環境変数や認証情報は決して記録しない)。

終了コード:
  0 = 報告の収集に成功し、ブロッカーなし (推論の正常動作を保証するもの
      ではない)
  1 = 不正な入力 (torch/HIP 不在、無効なデバイス指定、不正なスナップショット)
  2 = ブロッカー診断 (安全側の強制無効化、gfx1201 の LDS 期待値不一致、
      未対応アーキテクチャ)

詳細: doc/r9700_preflight.md
"""

import argparse
import json
import os
import re
import sys

SNAPSHOT_SCHEMA = "r9700-preflight-snapshot/v1"
REPORT_SCHEMA = "r9700-preflight-report/v1"
TOOL_NAME = "r9700_preflight"

# setup.py の SUPPORTED_GPU_ARCHS / GPU_ARCH_SMEM と対応する写し
# (setup.py を import せずに済むよう定数として持つ。ズレはテストで検証)。
SUPPORTED_GPU_ARCHS = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)

GPU_ARCH_SMEM = {
    "gfx1030": 65536,
    "gfx1100": 92160,
    "gfx1101": 92160,
    "gfx1102": 92160,
    "gfx1150": 65536,
    "gfx1151": 65536,
    "gfx1200": 92160,
    "gfx1201": 65536,
}
DEFAULT_ARCH_SMEM = 65536

R9700_ARCH = "gfx1201"
RDNA4_ARCHS = ("gfx1200", "gfx1201")
# doc/r9700_vs_v620.md で実測確認済みの gfx1201 LDS (shared_memory_per_block)
R9700_MEASURED_LDS_BYTES = 65536

FUSED_MOE_ENV = "EXL3_ROCM_RDNA4_FUSED_MOE"
BUILD_ARCH_ENV_VARS = ("PYTORCH_ROCM_ARCH", "GPU_ARCHS")

EXIT_OK = 0
EXIT_INVALID_INPUT = 1
EXIT_BLOCKED = 2


class PreflightError(Exception):
    """入力不正 (torch/HIP 不在、無効なデバイス、不正なスナップショット)。"""


def _env_on(name, default=False):
    """exllamav3/rocm_py/__init__.py の _env_on と同じ意味論を再現する。

    設定値の .strip() が ("", "0", "false", "False") のいずれでもなければ
    ON。"off" / "no" / "FALSE" のような文字列も ON になる点に注意。
    """
    if name not in os.environ:
        return default
    return os.environ[name].strip() not in ("", "0", "false", "False")


def normalize_arch(gcn_arch_name):
    """gcnArchName から機能サフィックスを取り除く (gfx1201:sramecc+:xnack- -> gfx1201)。"""
    if not isinstance(gcn_arch_name, str):
        return ""
    return gcn_arch_name.split(":", 1)[0].strip().lower()


def parse_build_archs(value):
    """PYTORCH_ROCM_ARCH / GPU_ARCHS の値を正規化アーキ名リストへ変換する。

    setup.py の _resolve_offload_archs と同様、カンマ区切り・空白区切りの
    両方を受け付け、各要素の機能サフィックスも除去する。
    """
    if not isinstance(value, str):
        return []
    parts = re.split(r"[,\s]+", value.strip())
    return [a for a in (normalize_arch(p) for p in parts) if a]


def collect_env():
    """ビルド/fused-MoE 関連の環境変数のみを記録する (他は一切記録しない)。"""
    env = {}
    for name in BUILD_ARCH_ENV_VARS:
        env[name] = os.environ.get(name)
    env[FUSED_MOE_ENV] = {
        "raw": os.environ.get(FUSED_MOE_ENV),
        "on": _env_on(FUSED_MOE_ENV, False),
    }
    return env


def load_torch():
    """torch を遅延 import する (テストでは差し替え可能)。"""
    import torch  # noqa: F401  (遅延 import: モジュール読込時は torch 不要)
    return torch


def collect_device_facts(device_index):
    """torch.cuda のデバイスプロパティから事実を収集する (読み取りのみ)。"""
    try:
        torch = load_torch()
    except ImportError as exc:
        raise PreflightError(
            "PyTorch を import できません (torch/HIP が未インストール): %s" % exc
        )

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if hip_version is None:
        raise PreflightError(
            "この torch ビルドは HIP バージョンを持ちません "
            "(torch.version.hip が None)。ROCm ビルドではないため GPU を照会できません。"
        )

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        raise PreflightError("torch.cuda が存在しません (この torch ビルドは CUDA/HIP をサポートしていません)。")

    # CUDA/HIP API 呼び出しは RuntimeError を投げ得る (HIP ランタイム不在、
    # ドライバー不整合など)。トレースバックを露出せず PreflightError に変換する。
    try:
        available = cuda.is_available()
    except RuntimeError as exc:
        raise PreflightError(
            "torch.cuda.is_available() が RuntimeError を投げました "
            "(HIP ランタイム/ドライバーを初期化できません): %s" % exc
        )
    if not available:
        raise PreflightError(
            "torch.cuda.is_available() が False です (HIP ランタイム/ドライバーが"
            "見えないか、利用可能な GPU デバイスがありません)。"
        )

    try:
        count = cuda.device_count()
    except RuntimeError as exc:
        raise PreflightError(
            "torch.cuda.device_count() が RuntimeError を投げました: %s" % exc
        )
    if device_index < 0 or device_index >= count:
        raise PreflightError(
            "デバイス番号 %d は範囲外です (検出されたデバイス数: %d)。"
            % (device_index, count)
        )

    try:
        props = cuda.get_device_properties(device_index)
    except RuntimeError as exc:
        raise PreflightError(
            "torch.cuda.get_device_properties(%d) が RuntimeError を投げました "
            "(デバイスを照会できません): %s" % (device_index, exc)
        )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "torch_version": getattr(torch, "__version__", None),
        "hip_version": hip_version,
        "device": {
            "index": device_index,
            "name": getattr(props, "name", None),
            "gcn_arch_name": getattr(props, "gcnArchName", None),
            "total_memory_bytes": getattr(props, "total_memory", None),
            "shared_memory_per_block": getattr(
                props, "shared_memory_per_block", None
            ),
        },
    }


def validate_snapshot(obj, origin="snapshot"):
    """スナップショット辞書のスキーマを検証し、事実辞書を返す。

    不正なら PreflightError。未知の追加キーは将来互換のため許容する。
    """
    if not isinstance(obj, dict):
        raise PreflightError("%s: JSON オブジェクトではありません。" % origin)

    schema = obj.get("schema")
    if schema != SNAPSHOT_SCHEMA:
        raise PreflightError(
            "%s: schema が「%s」ではありません (実際: %r)。"
            % (origin, SNAPSHOT_SCHEMA, schema)
        )

    for key in ("torch_version", "device"):
        if key not in obj:
            raise PreflightError("%s: 必須キー「%s」がありません。" % (origin, key))

    torch_version = obj["torch_version"]
    if not isinstance(torch_version, str) or not torch_version.strip():
        raise PreflightError("%s: torch_version は空でない文字列。" % origin)

    hip_version = obj.get("hip_version")
    if hip_version is not None and (not isinstance(hip_version, str) or not hip_version.strip()):
        raise PreflightError("%s: hip_version は null または空でない文字列。" % origin)

    dev = obj["device"]
    if not isinstance(dev, dict):
        raise PreflightError("%s: device はオブジェクト。" % origin)

    index = dev.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise PreflightError("%s: device.index は 0 以上の整数。" % origin)

    name = dev.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PreflightError("%s: device.name は空でない文字列。" % origin)

    gcn = dev.get("gcn_arch_name")
    if not isinstance(gcn, str) or not gcn.strip():
        raise PreflightError("%s: device.gcn_arch_name は空でない文字列。" % origin)

    mem = dev.get("total_memory_bytes")
    if not isinstance(mem, int) or isinstance(mem, bool) or mem <= 0:
        raise PreflightError("%s: device.total_memory_bytes は正の整数。" % origin)

    smem = dev.get("shared_memory_per_block")
    if smem is not None and (not isinstance(smem, int) or isinstance(smem, bool) or smem <= 0):
        raise PreflightError(
            "%s: device.shared_memory_per_block は正の整数または null。" % origin
        )

    return {
        "schema": SNAPSHOT_SCHEMA,
        "torch_version": torch_version,
        "hip_version": hip_version,
        "device": {
            "index": index,
            "name": name,
            "gcn_arch_name": gcn,
            "total_memory_bytes": mem,
            "shared_memory_per_block": smem,
        },
    }


def load_snapshot(path):
    """JSON スナップショットファイルを読み、検証済みの事実辞書を返す。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except UnicodeDecodeError as exc:
        # UTF-8 以外 (バイナリ等) は OSError でなく ValueError 系なので個別に扱う
        raise PreflightError(
            "スナップショットを UTF-8 でデコードできません (%s): %s" % (path, exc)
        )
    except OSError as exc:
        raise PreflightError("スナップショットを読み込めません (%s): %s" % (path, exc))
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise PreflightError("スナップショットが JSON として解析できません (%s): %s" % (path, exc))
    return validate_snapshot(obj, origin=path)


def _finding(findings, severity, code, message):
    findings.append({"severity": severity, "code": code, "message": message})


def analyze(facts, source, env):
    """事実と環境記録からレポート辞書を返す (ステータスと終了コードを含む)。"""
    findings = []
    next_steps = []
    blockers = []

    dev = facts["device"]
    arch = normalize_arch(dev["gcn_arch_name"])
    smem = dev["shared_memory_per_block"]
    expected_smem = GPU_ARCH_SMEM.get(arch, DEFAULT_ARCH_SMEM)

    # --- アーキテクチャ ---
    if arch not in SUPPORTED_GPU_ARCHS:
        _finding(
            findings,
            "error",
            "arch_unsupported",
            "正規化アーキ %s は setup.py の SUPPORTED_GPU_ARCHS に含まれません。"
            "この環境での EXL3 ビルド/ロードはサポート対象外です。" % arch,
        )
        blockers.append("arch_unsupported")
        status = "unsupported_arch"
    elif arch == R9700_ARCH:
        status = None  # ブロッカーがなければ comparison_workarounds_required
        _finding(
            findings,
            "info",
            "r9700_detected",
            "デバイス %s (%s) は R9700 と同じ gfx1201 アーキです。" % (dev["name"], dev["gcn_arch_name"]),
        )
    else:
        status = "not_r9700"
        _finding(
            findings,
            "info",
            "arch_not_r9700",
            "正規化アーキ %s はビルドターゲット一覧に含まれますが、gfx1201 ではありません。"
            "一覧への掲載は推論動作の検証を意味しません。" % arch,
        )
        if arch in RDNA4_ARCHS:
            # SUPPORTED_GPU_ARCHS  membership はビルドターゲット一覧にすぎない。
            # gfx1200 の推論実行は未検証で、WMMA 注意点は gfx1201 と同様に残る。
            _finding(
                findings,
                "warning",
                "rdna4_wmma_caveats",
                "arch %s は RDNA4 (gfx12xx) です。SUPPORTED_GPU_ARCHS への掲載は"
                "ビルドターゲットとしての一覧であり、この環境での推論実行が検証済み"
                "という意味ではありません。現行コードの WMMA パスは gfx12 用エンコーディング"
                "を持たず __builtin_trap() に到達し得るため、未検証のパスを有効化しないで"
                "ください (exllamav3/rocm_py の安全側の方向付けを参照)。",
            )

    # --- HIP バージョン (必須の事実) ---
    # ライブ計測では hip_version=None は収集前に PreflightError になるため、
    # ここに到達するのはスナップショットモード (null が保存された場合) のみ。
    if facts.get("hip_version") is None:
        _finding(
            findings,
            "error",
            "hip_missing",
            "HIP バージョンが不明です (hip_version が null)。ROCm/HIP バージョンが"
            "ないとビルド・互換性の判断ができないため、ブロッカーとして扱います。"
            "ライブ計測 (python3 rocm_tools/r9700_preflight.py) で torch から"
            "HIP バージョンを取得できるか確認してください。",
        )
        blockers.append("hip_missing")

    # --- LDS (shared_memory_per_block) ---
    if smem is None:
        _finding(
            findings,
            "warning",
            "lds_unknown",
            "torch のデバイスプロパティに shared_memory_per_block が報告されておらず、"
            "LDS を検証できません。",
        )
    elif arch == R9700_ARCH:
        if smem != R9700_MEASURED_LDS_BYTES:
            _finding(
                findings,
                "error",
                "lds_mismatch_r9700",
                "gfx1201 の実測 LDS 期待値は %d バイト (64KiB) ですが、%d が報告されました。"
                "fork が 92160 を仮定していた際は coop オートチューンで invalid argument "
                "が発生しました (doc/r9700_vs_v620.md)。この LDS 値ではビルド・ロードを"
                "進めるべきではありません。"
                % (R9700_MEASURED_LDS_BYTES, smem),
            )
            blockers.append("lds_mismatch_r9700")
        else:
            _finding(
                findings,
                "info",
                "lds_ok_r9700",
                "shared_memory_per_block = %d バイト (64KiB) は gfx1201 の実測期待値と一致します。"
                % R9700_MEASURED_LDS_BYTES,
            )
    elif smem != expected_smem:
        _finding(
            findings,
            "warning",
            "lds_unexpected",
            "shared_memory_per_block = %d はテーブル上の期待値 %d (arch %s) と異なります。"
            % (smem, expected_smem, arch),
        )

    # --- fused-MoE 強制 (exllamav3 の安全側の方向付けを無効化する) ---
    fused = env[FUSED_MOE_ENV]
    if fused["on"] and arch in RDNA4_ARCHS:
        _finding(
            findings,
            "error",
            "fused_moe_forced",
            "%s=%r が設定されています。gfx120x ではこの強制により fused_mode_buffers "
            "をクリアする安全側の方向付けが無効化され、WMMA に gfx12 のエンコーディングが"
            "無いため __builtin_trap() に到達し得ます (rdna_wmma.hip.h)。強制を解除"
            "してください。"
            % (FUSED_MOE_ENV, fused["raw"]),
        )
        blockers.append("fused_moe_forced")
    elif fused["on"]:
        _finding(
            findings,
            "warning",
            "fused_moe_forced_non_rdna4",
            "%s=%r が設定されていますが、このデバイス (arch %s) では gfx120x 向けの"
            "方向付けには影響しません。意味のある設定ではありません。"
            % (FUSED_MOE_ENV, fused["raw"], arch),
        )

    # --- ビルドターゲット環境変数 (将来のビルドに効くだけで、コンパ済み
    #     バイナリのアーキを実証するものではない) ---
    build_targets = {}
    for name in BUILD_ARCH_ENV_VARS:
        raw = env[name]
        if raw:
            build_targets[name] = {
                "raw": raw,
                "archs": parse_build_archs(raw),
            }
    effective_target = next((name for name in BUILD_ARCH_ENV_VARS if env[name]), None)
    for name, info in build_targets.items():
        info["effective"] = name == effective_target
    if not build_targets:
        _finding(
            findings,
            "info",
            "build_env_missing",
            "PYTORCH_ROCM_ARCH / GPU_ARCHS が未設定です。ビルド時は rocminfo からの"
            "自動解決になりますが、このデバイス向けに明示的にターゲット指定することを"
            "推奨します。",
        )
    else:
        for name, info in build_targets.items():
            if info["effective"] and arch and arch not in info["archs"]:
                _finding(
                    findings,
                    "warning",
                    "build_env_mismatch",
                    "%s=%r はこのデバイスのアーキ %s を含みません。この環境変数で"
                    "ビルドするとこのデバイスでロードできないバイナリが生成され得ます。"
                    % (name, info["raw"], arch),
                )

    _finding(
        findings,
        "info",
        "binary_target_unknown",
        "環境変数は今後のビルドにしか効かず、インストール済み EXL3 バイナリがどの"
        "アーキでコンパイルされたかの証拠にはなりません。コンパ済みバイナリの"
        "ターゲット検証はこのツールの範囲外であり unknown のままです。",
    )

    # --- ステータスと終了コード ---
    if status == "unsupported_arch":
        # 非対応アーキはステータスを維持したまま ブロック扱い (exit 2)
        exit_code = EXIT_BLOCKED
    elif blockers:
        status = "blocked"
        exit_code = EXIT_BLOCKED
    elif status is None:
        status = "comparison_workarounds_required"
        exit_code = EXIT_OK
    else:
        exit_code = EXIT_OK

    # --- 次のステップ (実測・ドキュメントに基づくもののみ) ---
    if arch == R9700_ARCH:
        next_steps.append(
            "ビルド前に PYTORCH_ROCM_ARCH=gfx1201 (または GPU_ARCHS=gfx1201) を"
            "明示して setup.py のターゲット解決と一致させてください。環境変数は"
            "新規ビルドにのみ効きます。"
        )
        next_steps.append(
            "gfx1201 の shared_memory_per_block は 65536 バイト (64KiB) が実測値です。"
            "92160 を仮定した形状選択は coop オートチューンで invalid argument を"
            "起こした実績があり、修正 (commit f2bd61b) は恒久的に入っています。"
        )
        next_steps.append(
            "EXL3_ROCM_RDNA4_FUSED_MOE は設定しないでください。gfx120x の fused-MoE "
            "は未対応で、強制すると WMMA の __builtin_trap() に到達し得ます。"
            "gfx12 WMMA パスの実装と実機検証が完了した構成でのみ強制を選択してください。"
        )
        next_steps.append(
            "R9700 と V620 の比較結果は外部アダプタ (r9700_balanced_run.py 等、"
            "本リポジトリ同梱外) を使ったものです。注意点と比較用ワークアラウンドは "
            "doc/r9700_vs_v620.md を参照してください。V620 専用のチューニングは "
            "R9700 に適用しないでください。"
        )
    elif arch in RDNA4_ARCHS:
        next_steps.append(
            "この RDNA4 デバイスの推論経路は未検証です。WMMA の既知の制約と "
            "doc/r9700_vs_v620.md を確認してください。gfx1201 の測定結果をそのまま適用しないでください。"
        )
    elif arch in SUPPORTED_GPU_ARCHS:
        next_steps.append(
            "このデバイス (%s) は R9700 ではないため、gfx1201 向けの LDS 期待値や"
            "比較用ワークアラウンドは適用されません。通常のビルド手順を参照してください。"
            % arch
        )
    else:
        next_steps.append(
            "このアーキ (%s) は setup.py のサポート対象外です。サポート対象 GPU を"
            "使うか、ビルドを自分で拡張する必要があります。" % arch
        )

    return {
        "tool": TOOL_NAME,
        "schema": REPORT_SCHEMA,
        "source": source,
        "status": status,
        "exit_code": exit_code,
        "torch_version": facts["torch_version"],
        "hip_version": facts["hip_version"],
        "device": dev,
        "normalized_arch": arch,
        "expected_smem_bytes": expected_smem,
        "env": env,
        "build_targets": build_targets,
        "binary_target": {
            "status": "unknown",
            "note": "環境変数はビルド意図の記録にすぎず、コンパ済みバイナリのアーキは未検証",
        },
        "findings": findings,
        "next_steps": next_steps,
    }


def render_text(report):
    """人間可読のテキストレポートを返す。"""
    lines = []
    lines.append("R9700 preflight (読み取り専用環境診断)")
    lines.append("ソース: %s" % report["source"])
    lines.append("ステータス: %s (exit %d)" % (report["status"], report["exit_code"]))
    lines.append("")
    lines.append("torch: %s / HIP: %s" % (report["torch_version"], report["hip_version"]))
    dev = report["device"]
    lines.append("デバイス %s: %s" % (dev["index"], dev["name"]))
    lines.append("  gcnArchName: %s -> 正規化アーキ: %s" % (dev["gcn_arch_name"], report["normalized_arch"]))
    lines.append("  VRAM: %s バイト" % dev["total_memory_bytes"])
    lines.append(
        "  shared_memory_per_block: %s (テーブル期待値: %s)"
        % (dev["shared_memory_per_block"], report["expected_smem_bytes"])
    )
    lines.append("")
    lines.append("記録した環境変数 (ビルド/fused-MoE 関連のみ):")
    for name in BUILD_ARCH_ENV_VARS:
        lines.append("  %s = %r" % (name, report["env"][name]))
    fused = report["env"][FUSED_MOE_ENV]
    lines.append("  %s = %r (ON: %s)" % (FUSED_MOE_ENV, fused["raw"], fused["on"]))
    lines.append("")
    lines.append("検査結果:")
    for f in report["findings"]:
        lines.append("  [%s] %s" % (f["severity"], f["message"]))
    lines.append("")
    lines.append("コンパ済みバイナリのターゲット: %s" % report["binary_target"]["status"])
    lines.append("")
    lines.append("次のステップ:")
    for step in report["next_steps"]:
        lines.append("  - %s" % step)
    lines.append("")
    lines.append(
        "注: exit 0 は事実の収集と診断の完了を意味します。推論の正常動作や"
        "汎用サポートを保証するものではありません。"
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="r9700_preflight.py",
        description=(
            "R9700 (gfx1201) 環境の読み取り専用プリフライト診断。"
            "モデルやネイティブ拡張はロードせず、ファイル変更もしない。"
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="照会するデバイス番号 (デフォルト: 0)。--snapshot 指定時はスナップショット内の index と照合。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="レポートを JSON で出力する",
    )
    parser.add_argument(
        "--snapshot",
        metavar="FILE",
        default=None,
        help="実機の代わりに JSON スナップショット (スキーマ: %s) から事実を読み込む" % SNAPSHOT_SCHEMA,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    as_json = args.json

    def emit_error(message):
        if as_json:
            print(json.dumps(
                {"tool": TOOL_NAME, "schema": REPORT_SCHEMA, "status": "error", "message": message},
                indent=2,
                ensure_ascii=False,
            ))
        else:
            print("エラー: %s" % message, file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        if args.snapshot is not None:
            facts = load_snapshot(args.snapshot)
            source = "snapshot:%s" % args.snapshot
            requested = args.device
            if requested is None:
                requested = facts["device"]["index"]
            if requested != facts["device"]["index"]:
                raise PreflightError(
                    "--device %d を要求しましたが、スナップショットはデバイス %d の"
                    "事実を記録しています。"
                    % (requested, facts["device"]["index"])
                )
        else:
            device_index = args.device if args.device is not None else 0
            if device_index < 0:
                raise PreflightError("--device は 0 以上の整数を指定してください (指定: %d)。" % device_index)
            facts = collect_device_facts(device_index)
            source = "device:%d" % device_index
        report = analyze(facts, source, collect_env())
    except PreflightError as exc:
        return emit_error(str(exc))

    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text(report))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())