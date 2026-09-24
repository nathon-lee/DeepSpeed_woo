#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Run paired real-kernel CUDA tests; preserve every group and both rounds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

MAIN = [(1024, 1024), (4096, 4096), (4096, 11008), (11008, 4096)]
PRESSURE = [(128, 8192), (8192, 128), (128, 16384), (16384, 128), (192, 5472), (5472, 192),
            (256, 4096), (4096, 256), (384, 2736), (2736, 384), (512, 2048), (2048, 512),
            (768, 1376), (1376, 768), (1024, 1024)]
TRANSFORM = [(1, 128, 12, 64), (1, 2048, 32, 128), (1, 2048, 28, 128), (8, 512, 32, 128)]
BASE = '0b987c0e99f546e7867f0c6af4c8336d7d1e2eca'
TARGET = 'e90e6aef4a17a08446dd362aeed54042387bd531'


def command(args):
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {'returncode': p.returncode, 'output': p.stdout}
    except OSError as exc:
        return {'returncode': None, 'output': str(exc)}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, tag, directory):
    from torch.utils.cpp_extension import load
    build_dir = directory / tag
    build_dir.mkdir()
    flags = ['-O3', '-DHALF_PRECISION_AVAILABLE', '-U__CUDA_NO_HALF_OPERATORS__',
             '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '--ptxas-options=-v']
    save(build_dir / 'inputs.json', {'root': str(root), 'source_sha256': digest(root / 'csrc/transformer/transform_kernels.cu'),
                                   'cuda_flags': flags, 'cflags': ['-O3'], 'ldflags': ['-lcurand', '-lcublas', '-Wl,-Bsymbolic']})
    return load(name='transpose_' + tag, sources=[str(Path(__file__).with_name('bindings.cpp')),
                str(root / 'csrc/transformer/transform_kernels.cu')],
                extra_include_paths=[str(root / 'csrc/includes')], extra_cflags=['-O3'],
                extra_cuda_cflags=flags, extra_ldflags=['-lcurand', '-lcublas', '-Wl,-Bsymbolic'],
                build_directory=str(build_dir), verbose=True)


def sass(module, directory, tag):
    # Hash the selected function body, excluding ELF/module file headers.
    result = command(['cuobjdump', '-sass', '-arch', 'sm_90', module.__file__])
    (directory / (tag + '.sass')).write_text(result['output'])
    if result['returncode'] != 0:
        raise RuntimeError('cuobjdump failed')
    hashes = {}
    for section in re.split(r'Function : ', result['output'])[1:]:
        name = section.splitlines()[0].strip()
        demangled = command(['c++filt', name])['output'].strip()
        if 'void transform_0213<' not in demangled and 'void Transpose_Kernel<' not in demangled:
            continue
        instructions = '\n'.join(line.strip() for line in section.splitlines()[1:] if '/*' in line)
        if not instructions:
            raise RuntimeError('empty SASS for ' + name)
        hashes[name] = hashlib.sha256(instructions.encode()).hexdigest()
    if len(hashes) != 4:
        raise RuntimeError('expected exactly four fallback kernel SASS bodies')
    save(directory / (tag + '_sass_hashes.json'), hashes)
    return hashes


def suite(modules, directory, torch, np):
    cuda = torch.cuda  #ignore-cuda
    failures = []

    def check(shape, dtype, offset=0, transform=False, baseline=True, timing=True):
        n = int(np.prod(shape))
        x = torch.empty(n + offset, device='cuda', dtype=dtype)[offset:].view(shape)
        x.uniform_(-1, 1)
        reference = x.permute(0, 2, 1, 3).contiguous() if transform else x.t().contiguous()
        outputs = {tag: torch.empty(n + offset, device='cuda', dtype=dtype)[offset:].view(reference.shape)
                   for tag in (('baseline', 'candidate') if baseline else ('candidate',))}
        row = {'shape': shape, 'dtype': str(dtype), 'offset': offset, 'input_mod16': x.data_ptr() % 16,
               'output_mod16': {k: v.data_ptr() % 16 for k, v in outputs.items()}, 'correctness': {}}
        for tag, y in outputs.items():
            modules[tag].run(x, y, transform)
            cuda.synchronize()
            exact = torch.equal(y.view(torch.uint8), reference.view(torch.uint8))
            row['correctness'][tag] = {'bitwise_exact': exact, 'finite': bool(y.isfinite().all()),
                                       'max_abs_error': float((y.float() - reference.float()).abs().max())}
            if not exact or not row['correctness'][tag]['finite']:
                failures.append({'kind': 'correctness', 'row': row})
        if timing:
            for _ in range(30):
                for tag, y in outputs.items():
                    modules[tag].run(x, y, transform)
            cuda.synchronize()
            samples = {tag: [] for tag in outputs}
            start, stop = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
            for group in range(60):
                order = list(outputs) if group % 2 == 0 else list(reversed(outputs))
                for tag in order:
                    start.record()
                    for _ in range(100):
                        modules[tag].run(x, outputs[tag], transform)
                    stop.record()
                    stop.synchronize()
                    samples[tag].append(start.elapsed_time(stop) / 100)
            row['group_ms'] = samples
            for tag, values in samples.items():
                row[tag] = dict(zip(['p10_ms', 'median_ms', 'p90_ms'], map(float, np.percentile(values, [10, 50, 90]))))
            row['speedup_pct'] = (row['baseline']['median_ms'] / row['candidate']['median_ms'] - 1) * 100
            if offset == 0:
                row['margin_pass'] = row['speedup_pct'] >= 10 and row['candidate']['p90_ms'] < row['baseline']['p10_ms']
                if not row['margin_pass']:
                    failures.append({'kind': 'performance_margin', 'row': row})
        return row

    for round_id in (1, 2):
        rows = []
        for dtype in (torch.float16, torch.float32):
            for shape in MAIN:
                rows.append(check(shape, dtype))
                save(directory / f'main_round{round_id}.json', rows)
    for name, shapes, options in [('pressure', PRESSURE, {}),
                                  ('boundary', [(33, 64), (33, 65)], {'baseline': False, 'timing': False}),
                                  ('misaligned', [(1024, 1024)], {'offset': 1}),
                                  ('transform_0213', TRANSFORM, {'transform': True, 'timing': False})]:
        rows = []
        for dtype in (torch.float16, torch.float32):
            for shape in shapes:
                rows.append(check(shape, dtype, **options))
                save(directory / (name + '.json'), rows)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    query = ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits']
    metadata = {'timestamp': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv, 'hostname': platform.node(),
                'baseline_expected': BASE, 'target_expected': TARGET,
                'gpu_before': command(query), 'driver': command(['nvidia-smi']),
                'nvcc': command(['nvcc', '--version']), 'compiler': command(['c++', '--version']),
                'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES'), 'status': 'running'}
    for tag in ('baseline', 'candidate'):
        root = getattr(args, tag).resolve()
        setattr(args, tag, root)
        metadata[tag] = {'root': str(root), 'head': command(['git', '-C', str(root), 'rev-parse', 'HEAD']),
                         'status': command(['git', '-C', str(root), 'status', '--short'])}
    save(args.output / 'environment.json', metadata)
    try:
        import torch
        import numpy as np
        cuda = torch.cuda  #ignore-cuda
        metadata.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=cuda.is_available())
        if not cuda.is_available() or not shutil.which('nvcc'):
            metadata.update(status='blocked', reason='CUDA GPU/toolchain unavailable; no kernels built or measured')
            return 2
        # Require an explicitly selected idle device. Check before creating tensors.
        if not os.getenv('CUDA_VISIBLE_DEVICES') or cuda.device_count() != 1:
            raise RuntimeError('select exactly one idle GPU with CUDA_VISIBLE_DEVICES')
        metadata['selected_gpu'] = cuda.get_device_name(0)
        order = ['candidate', 'baseline'] if args.reverse else ['baseline', 'candidate']
        modules = {tag: build(getattr(args, tag), tag, args.output) for tag in order}
        hashes = {tag: sass(module, args.output, tag) for tag, module in modules.items()}
        metadata['sass_equal'] = hashes['baseline'] == hashes['candidate']
        failures = suite(modules, args.output, torch, np)
        metadata['failures'] = failures
        metadata['status'] = 'passed' if not failures and metadata['sass_equal'] else 'failed'
        return 0 if metadata['status'] == 'passed' else 1
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        metadata['gpu_after'] = command(query)
        save(args.output / 'environment.json', metadata)
        print(metadata['status'], args.output, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
