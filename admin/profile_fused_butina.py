#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import torch
from rdkit import Chem

import nvmolkit

nvmolkit.__path__.append("/home/mastrmatt/nvMolKit/build/nvmolkit")

from nvmolkit.clustering import fused_butina
from nvmolkit.fingerprints import MorganFingerprintGenerator


def load_molecules(path: str, count: int):
    molecules = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            mol = Chem.MolFromSmiles(line.split()[0])
            if mol is not None:
                molecules.append(mol)
            if len(molecules) == count:
                break
    if len(molecules) != count:
        raise RuntimeError(f"Only loaded {len(molecules)} molecules, requested {count}")
    return molecules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9000)
    parser.add_argument("--cutoff", type=float, default=0.35)
    parser.add_argument("--metric", default="tanimoto", choices=["tanimoto", "cosine"])
    parser.add_argument("--smiles", default="benchmarks/data/chembl_10k.smi")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--capture-range", action="store_true")
    args = parser.parse_args()

    molecules = load_molecules(args.smiles, args.size)
    fps = MorganFingerprintGenerator(radius=2, fpSize=1024).GetFingerprints(molecules, 10).torch()
    torch.cuda.synchronize()

    if args.warmup:
        fused_butina(fps, cutoff=args.cutoff, metric=args.metric)
        torch.cuda.synchronize()

    if args.capture_range:
        torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_fused_butina")
    start = time.perf_counter()
    clusters, cluster_sizes = fused_butina(fps, cutoff=args.cutoff, metric=args.metric)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    torch.cuda.nvtx.range_pop()
    if args.capture_range:
        torch.cuda.cudart().cudaProfilerStop()

    print(
        f"size={args.size} cutoff={args.cutoff} metric={args.metric} "
        f"clusters={len(clusters)} assigned={cluster_sizes[-1]} elapsed_ms={elapsed_ms:.3f}"
    )


if __name__ == "__main__":
    main()
