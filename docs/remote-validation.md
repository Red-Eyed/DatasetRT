# Run benchmarks on another machine

Install a wheel containing this feature with its `benchmark` extra, into an
environment with the correct PyTorch build for that machine. For example, use
`uv pip install '/path/to/dataset_rt-wheel.whl[benchmark]'`, substituting the
actual wheel filename. Select CUDA-enabled PyTorch using the official
[installation instructions](https://pytorch.org/get-started/locally/).
The current development changes must be present in the installed wheel;
an older published release does not contain this entry point.

From an already prepared project environment, the equivalent command prefix is
`uv run --python 3.11 --extra benchmark python`. The commands below use that prefix.
The runner supports Linux and macOS. CUDA/NCCL requires appropriate GPU hardware;
requesting CUDA fails rather than switching silently to CPU.

## One GPU and local DDP

Start with one GPU and no loader workers to obtain a comparison baseline:

```bash
uv run --python 3.11 --extra benchmark python -m dataset_rt.run_benchmark \
  --device cuda --ranks 1 --num-workers 0 \
  --samples-per-cache 8192 --caches 2 --batch-size 128 \
  --input-features 1024 --model-width 2048 --repeats 5 \
  --output results/gpu-1-workers-0.json
```

Then run four GPUs with eight workers per GPU:

```bash
uv run --python 3.11 --extra benchmark python -m dataset_rt.run_benchmark \
  --device cuda --ranks 4 --num-workers 8 --pin-memory \
  --samples-per-cache 8192 --caches 2 --batch-size 128 \
  --input-features 1024 --model-width 2048 --repeats 5 \
  --output results/gpu-4-workers-8.json
```

`--ranks` counts training processes, with one visible GPU per local rank.
`--num-workers` counts loader workers **per rank**, so the second command creates
32 loader workers plus four training processes. Native threads and metadata
allocations also multiply across consumers. Repeat with workers 0, 1, 2, 4, and 8
using distinct output filenames. Compare worker counts at the same rank count
before interpreting distributed scaling; adding ranks changes the global batch.

The module automatically invokes `torchrun` with a private local rendezvous and
no automatic restarts. CUDA loading defaults to `spawn`; CUDA with fork workers
is rejected. `forkserver` is also available. These choices follow PyTorch's
[DDP multiprocessing guidance](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

## CPU and workload controls

This command verifies the portable CPU/Gloo path without GPU hardware:

```bash
uv run --python 3.11 --extra benchmark python -m dataset_rt.run_benchmark \
  --device cpu --ranks 2 --num-workers 2 --no-shuffle \
  --samples-per-cache 1000 --batch-size 32 --repeats 3 \
  --output results/cpu-ddp.json
```

On this Mac, set `GLOO_SOCKET_IFNAME=lo0` for local CPU communication if automatic
interface selection fails. The module does not force a network interface.

- `--payload-bytes` controls stored payload size; `--metadata-bytes` adds a wide
  metadata value in both population metadata and stored records. It measures the
  full read path, including record metadata, rather than isolating control IPC.
- `--transform-rounds` adds CPU hashing work per sample. Features are fixed-size
  tensors derived from payload hashes; this is not an image/audio decoder.
- `--input-features` (a multiple of 32) and `--model-width` control synthetic
  training compute. The model is two linear layers with a ReLU and SGD.
- `--samples-per-epoch` is global across ranks. By default it is the physical
  population size. Padding/discarding and warnings follow the loader contract.
- `--worker-partition split|replicate`, `--no-shuffle`, `--drop-last`, and
  `--no-persistent-workers` exercise the corresponding loader behavior.
- `--timeout-seconds` bounds the complete automatic local launch and distributed
  collectives. Increase it for long benchmarks. `--worker-timeout` bounds worker
  batch waits. `--quiet` suppresses harness progress, while errors remain visible.

For pure reader comparisons against direct native iteration, the existing
paired benchmark remains available from a checkout:

```bash
uv run --python 3.11 --extra dev scripts/bench_reader.py \
  --contexts '["spawn"]' --worker-counts '[1,2,4,8]' --repeats 5 \
  --output results/readers.json
```

It includes native and zero-worker controls automatically. Performance targets
are reported as measurements; hardware-dependent timings are not pytest assertions.

## Existing caches

```bash
uv run --python 3.11 --extra benchmark python -m dataset_rt.run_benchmark \
  --device cuda --ranks 2 --num-workers 4 --pin-memory \
  --cache-paths '["/data/cache-a","/data/cache-b"]' \
  --samples-per-epoch 64000 --batch-size 128 --repeats 5 \
  --output results/existing-caches.json
```

Existing caches are read without modification. Synthetic cache-size options
apply only when no cache paths are supplied. Payloads may contain arbitrary bytes:
the benchmark hashes them into model inputs. It reads original cache metadata
and default weights; it does not load a separately edited active metadata table.
Keep cache contents and ordering consistent across ranks. Fixture generation is
automatic and excluded from timed passes; synthetic fixtures are private copies
per rank, so their results do not measure contention on shared dataset files.

## Multiple hosts

Run one external launcher per host. Example for two hosts with four GPUs each;
replace `10.0.0.10` with the reachable address of host zero.

Host zero:

```bash
uv run --python 3.11 --extra benchmark python -m torch.distributed.run \
  --nnodes 2 --nproc-per-node 4 --node-rank 0 \
  --master-addr 10.0.0.10 --master-port 29500 --max-restarts 0 \
  --module dataset_rt.run_benchmark \
  --device cuda --num-workers 8 --pin-memory \
  --samples-per-epoch 64000 --batch-size 128 --repeats 5 \
  --output results/multihost.json
```

Host one runs the same command with `--node-rank 1`. Do not add `--ranks`:
the module uses the launcher's `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` and does not
launch nested ranks. Global rank zero writes the complete JSON report. In this
example there are eight training ranks and 64 loader workers globally.

Use identical code, benchmark settings, and active metadata on all hosts. With
existing-cache mode, every host needs the same caches in the same order; local
paths may differ. The runner fingerprints metadata, settings, and source code
and rejects disagreements before training. Network interfaces and connectivity
remain machine-specific; use [PyTorch's distributed configuration](https://docs.pytorch.org/docs/stable/distributed.html).
For external launches, the launcher/operator owns the overall job deadline;
the module's collective and worker timeouts still apply. Use fixed membership
and no automatic restarts for comparable benchmark runs.

## Read the results

The report records per-rank startup time, first-batch latency, batch-wait time,
training time, sample/batch counts, sampled process-tree RSS, threads, descriptors,
and CUDA peak allocated/reserved bytes. CPU runs report zero GPU allocation.
It includes hardware names, Python/PyTorch/package versions, code fingerprints,
settings, and verified replay/model-agreement results.

Global samples/second uses the total actual samples divided by the slowest rank's
elapsed time. Requested and effective budgets are both recorded. The first pass
includes reader/worker startup; subsequent passes use persistent workers by
default. This does not establish a cold filesystem cache. RSS sampling can miss
short peaks and counts shared pages repeatedly. CUDA values measure PyTorch
allocator memory, not all driver/NCCL memory. Constructor-export memory is outside
the per-pass observer. CUDA steps synchronize individually for bounded, completed
wall-clock measurements, so these are not overlapped production-training timings.

Checks fail the run for incorrect counts, plan coverage, sequential synthetic
positions, seeded replay, or model divergence. Replay checks at most four batches;
it is not a proof of full-epoch distribution. Shuffled reads remain weighted
sampling with replacement, and split quotas need not match global weight sums.

Run the real harness tests on the target machine:

```bash
uv run --python 3.11 --extra dev pytest \
  tests/integrations/test_benchmark_module.py -m slow -n 0 -v
```

To run only actual GPU checks, add `-k cuda -rs`. These tests skip if the required
CUDA/NCCL build or one/two visible GPUs are missing. CPU tests are runnable here;
CUDA/NCCL and physical multi-host performance remain unverified until executed
on suitable machines. Save the JSON files for comparison across settings/revisions.
