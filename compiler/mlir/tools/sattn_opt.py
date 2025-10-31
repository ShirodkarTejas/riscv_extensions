#!/usr/bin/env python3
"""
Lightweight MLIR-like pipeline tool for the SATTN dialect.
This does not depend on MLIR; it performs simple text rewrites to emulate:
  - materialize-indices
  - fuse-softmax
  - lower-to-{rvv|rocc}

Usage:
  python compiler/mlir/tools/sattn_opt.py --passes materialize-indices,fuse-softmax,lower-to-rvv \
    --in compiler/mlir/examples/sattn_example.mlir --out /tmp/out.mlir
"""
import argparse


def pass_materialize_indices(text: str) -> str:
    # Add an attribute materialized_indices=true if not present
    if 'materialized_indices' in text:
        return text
    return text.replace('"sattn.sparse_attention"(', '"sattn.sparse_attention"(').replace(
        'softmax_mode = "', 'materialized_indices = true, softmax_mode = "'
    )


def pass_fuse_softmax(text: str) -> str:
    # If softmax_mode is logsumexp, tag fused_softmax=true
    if 'fused_softmax' in text:
        return text
    return text.replace('softmax_mode = "logsumexp"', 'softmax_mode = "logsumexp", fused_softmax = true')


def pass_lower_to_backend(text: str, backend: str) -> str:
    # Annotate with backend selection
    key = 'lowered_backend'
    if key in text:
        return text
    return text.replace('precision = "', f'lowered_backend = "{backend}", precision = "')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', type=str, required=True,
                    help='Comma-separated list: materialize-indices,fuse-softmax,lower-to-rvv|lower-to-rocc')
    ap.add_argument('--in', dest='inp', type=str, required=True)
    ap.add_argument('--out', dest='out', type=str, required=True)
    args = ap.parse_args()

    with open(args.inp, 'r') as f:
        text = f.read()

    for p in args.passes.split(','):
        p = p.strip()
        if p == 'materialize-indices':
            text = pass_materialize_indices(text)
        elif p == 'fuse-softmax':
            text = pass_fuse_softmax(text)
        elif p == 'lower-to-rvv':
            text = pass_lower_to_backend(text, 'rvv')
        elif p == 'lower-to-rocc':
            text = pass_lower_to_backend(text, 'rocc')
        else:
            raise SystemExit(f'Unknown pass: {p}')

    with open(args.out, 'w') as f:
        f.write(text)

    print(f'Wrote transformed MLIR to {args.out}')


if __name__ == '__main__':
    main()


