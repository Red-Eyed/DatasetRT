"""Small reader identity and row-partition helpers; no loader or native state."""

from __future__ import annotations

import secrets

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_rt.records import RowSpan

U64_MAX = (1 << 64) - 1


class ReplicaIdentity(BaseModel):
    """Captured data-parallel identity; loading workers never query process groups."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    rank: int = Field(default=0, ge=0, lt=1 << 32)
    world_size: int = Field(default=1, gt=0, le=1 << 32)

    @model_validator(mode="after")
    def _validate_membership(self) -> ReplicaIdentity:
        """Reject a rank outside its captured process group."""
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        return self


def _require_u64(name: str, value: int) -> None:
    """Reject coercions and values that cannot be encoded in seed identities."""
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} must be a u64 integer")


def _mix_u64(value: int) -> int:
    """Permute u64 values with the invertible SplitMix64 finalizer."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & U64_MAX
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & U64_MAX
    return value ^ (value >> 31)


def derive_seed(base_seed: int, rank_id: int, worker_id: int) -> int:
    """Mix a versioned tuple into a native seed, avoiding additive ID collisions.

    world_size is not an identity. This O(1) derivation is reproducible across
    processes, unlike Python hash(), clocks, PIDs, or inherited mutable RNGs.
    Pack u32 rank/worker IDs into a unique u64 slot, then apply a seed-keyed
    permutation. Distinct rank/worker pairs cannot collide for one base seed;
    that does not prove statistical independence of native random streams.
    """
    for name, value in (("base_seed", base_seed), ("rank_id", rank_id), ("worker_id", worker_id)):
        _require_u64(name, value)
    if rank_id >= 1 << 32 or worker_id >= 1 << 32:
        raise ValueError("rank_id and worker_id must fit u32")
    slot = (rank_id << 32) | worker_id
    # This fixed domain constant is part of reader seed derivation version 1.
    key = _mix_u64(base_seed ^ 0x4453545253454544)
    return _mix_u64(slot ^ key)


def reader_seed(*, shuffle: bool, seed: int | None, rank_id: int, worker_id: int) -> int:
    """Select a seed during process-local setup; sequential mode ignores it.

    None is the optional API input, converted here into fresh OS randomness.
    No random seed is generated during recipe creation in the parent process.
    """
    if not shuffle:
        return 0
    base_seed = secrets.randbits(64) if seed is None else seed
    return derive_seed(base_seed, rank_id, worker_id)


def partition_rows(count: int, *, parts: int, part_id: int) -> RowSpan:
    """Split a finite count into contiguous spans in O(1), including empty tails.

    First spans receive one extra row when division has a remainder. Splitting
    row positions preserves intentional duplicate physical sample identities.
    """
    _require_u64("count", count)
    _require_u64("parts", parts)
    _require_u64("part_id", part_id)
    if parts == 0 or part_id >= parts:
        raise ValueError("part_id must belong to a positive partition count")
    size, extra = divmod(count, parts)
    return RowSpan(part_id * size + min(part_id, extra), size + int(part_id < extra))


def worker_rows(
    count: int, replica: ReplicaIdentity, *, num_workers: int, worker_id: int
) -> RowSpan:
    """Split rank rows among actual local workers without depending on batch size.

    Zero DataLoader workers means one consuming iterator in the rank process.
    Different worker counts on different ranks still preserve exact coverage.
    """
    _require_u64("num_workers", num_workers)
    rank_span = partition_rows(count, parts=replica.world_size, part_id=replica.rank)
    local = partition_rows(rank_span.length, parts=max(1, num_workers), part_id=worker_id)
    return RowSpan(rank_span.offset + local.offset, local.length)


def worker_budgets(
    samples: int, *, workers: int, populations: int, batch_size: int
) -> tuple[int, ...]:
    """Assign whole batches and at most one tail without budgeting an empty population.

    Fewer nonempty populations than workers leaves idle workers. Keeping one tail makes
    Torch's ordinary length calculation exact for either drop_last setting.
    """
    if samples == 0:
        return (0,) * workers
    if populations == 0:
        raise ValueError("samples_per_epoch requires a nonempty population")
    batches, tail = divmod(samples, batch_size)
    active = min(workers, populations, max(1, batches))
    budgets = [
        partition_rows(batches, parts=active, part_id=i).length * batch_size for i in range(active)
    ]
    budgets[-1] += tail
    return tuple(budgets) + (0,) * (workers - active)


def capture_replica() -> ReplicaIdentity:
    """Capture initialized-group rank/size in the training process before workers.

    Torch remains optional until this helper is invoked. With no initialized
    distributed group, use a single replica. Hybrid subgroup selection requires
    separate explicit support; the default group is appropriate for plain DDP.
    """
    import torch.distributed as distributed
    from torch.utils.data import get_worker_info

    if get_worker_info() is not None:
        raise RuntimeError("capture_replica must run before DataLoader workers start")
    if not distributed.is_available() or not distributed.is_initialized():
        return ReplicaIdentity()
    return ReplicaIdentity(rank=distributed.get_rank(), world_size=distributed.get_world_size())
