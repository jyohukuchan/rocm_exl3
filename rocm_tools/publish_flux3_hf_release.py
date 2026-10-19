"""Publish a generation-validated FLUX.3 package and verify every remote file."""
import argparse
import json
from pathlib import Path

from rocm_tools.publish_qwen_image21_hf_release import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    validation = json.loads((args.folder / 'VALIDATION.json').read_text())
    assert validation['tensor_payload_files_verified_identical'] == 777
    staged = validation['staged_4bit_generation']
    assert staged['pixel_hash_matches_evaluated_4bit'] and staged['dense_dit_absent']
    assert staged['fresh_text_encoding'] and staged['active_exl3_layers'] == 228
    assert all(m['actual_steps'] == 16 and m['finite_final_latents'] and m['finite_vae_output']
               for m in staged['measurements'])
    publish(args)


if __name__ == '__main__':
    main()
