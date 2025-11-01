#!/usr/bin/env python3
import argparse, os, re, subprocess, sys, tempfile


def run(cmd):
    print('[run]', ' '.join(cmd))
    return subprocess.check_output(cmd, text=True)


def _capture_attrs(block):
    attrs = block
    def geti(name):
        mm = re.search(rf"{name}\s*=\s*([0-9]+)", attrs)
        return int(mm.group(1)) if mm else None
    def getf(name):
        mm = re.search(rf"{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)", attrs)
        return float(mm.group(1)) if mm else None
    def gets(name):
        mm = re.search(rf"{name}\s*=\s*\"([^\"]+)\"", attrs)
        return mm.group(1) if mm else None
    return {
        'spec': gets('spec') or 'sliding_window',
        'L': geti('s_tokens') or geti('tile_S') or 128,
        'D': geti('head_dim_d') or geti('tile_D') or 32,
        'window_size': geti('window_size') or 8,
        'block_size': geti('block_size') or 64,
        'dilation': geti('dilation') or 1,
        'wrap': geti('wrap') or 0,
        'keep_ratio': getf('keep_ratio') or 0.12,
        'global_tokens': geti('global_tokens') or 0,
        'nm_n': geti('nm_n') or 0,
        'nm_m': geti('nm_m') or 0,
        'lsh_buckets': geti('lsh_buckets') or 0,
        'num_landmarks': geti('num_landmarks') or 0,
        'landmark_iters': geti('landmark_iters') or 0,
        'gqa_group_size': geti('gqa_group_size') or 1,
        'comp_block_size': geti('comp_block_size') or 0,
        'precision': gets('precision') or 'fp32',
        'scale_q': getf('scale_q'),
        'scale_k': getf('scale_k'),
        'scale_v': getf('scale_v'),
    }

def parse_attrs(txt):
    m = re.search(r"sattn\.rvv_call[^\{]*\{([^}]*)\}", txt, re.MULTILINE | re.DOTALL)
    if m:
        return _capture_attrs(m.group(1))
    m2 = re.search(r"sattn\.sparse_attention[^\{]*\{([^}]*)\}", txt, re.MULTILINE | re.DOTALL)
    if m2:
        return _capture_attrs(m2.group(1))
    return {}


def main():
    ap = argparse.ArgumentParser(description='Run RVV kernel selected by MLIR spec')
    ap.add_argument('--mlir', required=True)
    ap.add_argument('--sattn-opt', default='build/mlir/tools/sattn-opt/sattn-opt')
    ap.add_argument('--runner', default='build/backends/rvv/sattn_rvv_runner')
    ap.add_argument('--autotune', action='store_true')
    ap.add_argument('--prefer-bsr', action='store_true', help='Hint selector to prefer block specs')
    ap.add_argument('--prefer-sw', action='store_true', help='Hint selector to prefer sliding-window')
    ap.add_argument('--l1-bytes', type=int, default=0, help='Override L1 size for selector cache-fit heuristic')
    ap.add_argument('--use-hw-probe', action='store_true', help='Probe RoCC sim caps to steer selector')
    args = ap.parse_args()

    lowered = args.mlir + '.rvv.mlir'
    try:
        if not os.path.exists(args.sattn_opt):
            raise FileNotFoundError(args.sattn_opt)
        # Set hardware preference envs during selection
        env = os.environ.copy()
        if args.prefer_bsr: env['SATTN_PREFER_BSR'] = '1'
        if args.prefer_sw: env['SATTN_PREFER_SW'] = '1'
        if args.l1_bytes and args.l1_bytes > 0: env['SATTN_HW_L1_BYTES'] = str(args.l1_bytes)
        if args.use_hw_probe:
            try:
                out = subprocess.check_output(['hw/sim/obj_dir/Vrocc_sattn'], text=True)
                for line in out.splitlines():
                    if line.startswith('rocc_hw:') and 'caps=' in line:
                        import re
                        m = re.search(r"caps=0x([0-9a-fA-F]+)", line)
                        if m:
                            caps = int(m.group(1), 16)
                            if (caps & 0x1) == 0: env['SATTN_DISABLE_BSR'] = '1'
                            if (caps & 0x2) == 0: env['SATTN_DISABLE_SW'] = '1'
                        break
            except Exception:
                pass
        print('[run]', args.sattn_opt, args.mlir, '--allow-unregistered-dialect', '-pass-pipeline=builtin.module(sattn-lower-rvv)')
        subprocess.check_call([args.sattn_opt, args.mlir, '--allow-unregistered-dialect', '-pass-pipeline=builtin.module(sattn-lower-rvv)'], env=env)
        # Since sattn-opt writes to stdout, capture and save
        out = subprocess.check_output([args.sattn_opt, args.mlir, '--allow-unregistered-dialect', '-pass-pipeline=builtin.module(sattn-lower-rvv)'], text=True, env=env)
        with open(lowered, 'w') as f: f.write(out)
        txt = out
    except Exception as e:
        print('[warn] sattn-opt unavailable or failed; using input MLIR directly:', e)
        txt = open(args.mlir).read()

    attrs = parse_attrs(txt)
    keep_x1000 = int(round((attrs['keep_ratio'] if attrs['keep_ratio'] is not None else 0.12) * 1000.0))
    cmd = [args.runner,
           '--spec', attrs['spec'], '--L', str(attrs['L']), '--D', str(attrs['D']),
           '--window', str(attrs['window_size']), '--block_size', str(attrs['block_size']),
           '--dilation', str(attrs['dilation']), '--wrap', str(attrs['wrap']),
           '--global_tokens', str(attrs['global_tokens']), '--nm_n', str(attrs['nm_n']), '--nm_m', str(attrs['nm_m']),
           '--lsh_buckets', str(attrs['lsh_buckets']), '--keep_x1000', str(keep_x1000)]
    # Optional group/compression knobs
    if attrs.get('gqa_group_size'):
        cmd += ['--gqa_group_size', str(attrs['gqa_group_size'])]
    if attrs.get('comp_block_size'):
        cmd += ['--comp_block_size', str(attrs['comp_block_size'])]

    # For block-based specs, optionally emit indices and pass to runner
    if attrs['spec'] in ('block_local_global', 'bsr', 'topk_per_query', 'nm_structured'):
        try:
            idx_path = os.path.splitext(args.mlir)[0] + '.indices.txt'
            out_lower = args.mlir  # we can reuse input for emitter
            run([sys.executable, 'compiler/mlir/tools/sattn_emit_indices_txt.py', '--in-mlir', out_lower, '--out-indices', idx_path])
            if os.path.exists(idx_path):
                cmd += ['--indices', idx_path]
        except Exception:
            pass
    if attrs['spec'] == 'landmark':
        if attrs.get('num_landmarks'):
            cmd += ['--landmarks', str(attrs['num_landmarks'])]
        if attrs.get('landmark_iters'):
            cmd += ['--landmark_iters', str(attrs['landmark_iters'])]
    # Precision/scales
    precision = attrs.get('precision') or 'fp32'
    if precision:
        cmd += ['--precision', precision]
    for key, flag in [('scale_q','--scale_q_x1000'), ('scale_k','--scale_k_x1000'), ('scale_v','--scale_v_x1000')]:
        if attrs.get(key) is not None:
            cmd += [flag, str(int(round(attrs[key] * 1000.0)))]

    # Fallback runner path if missing
    runner_path = cmd[0]
    if not os.path.exists(runner_path):
        alt = 'build/rvv/sattn_rvv_runner'
        if os.path.exists(alt):
            cmd[0] = alt

    if args.autotune:
      cmd.append('--autotune')
    out = run(cmd)
    print(out)


if __name__ == '__main__':
    main()


