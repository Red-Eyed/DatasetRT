use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::types::{CacheError, CacheResult};

pub enum EpochSampler {
    Uniform {
        rng: ChaCha8Rng,
        remaining: usize,
        sample_count: usize,
    },
    Weighted {
        distribution: WeightedIndex<f64>,
        rng: ChaCha8Rng,
        remaining: usize,
    },
}

impl EpochSampler {
    /// Create a uniform replacement sampler without allocating a full weight vector.
    pub fn uniform(sample_count: usize, seed: u64, epoch: u64) -> CacheResult<Self> {
        if sample_count == 0 {
            return Err(CacheError::InvalidInput(
                "cannot sample an empty epoch".to_string(),
            ));
        }
        Ok(Self::Uniform {
            rng: epoch_rng(seed, epoch),
            remaining: sample_count,
            sample_count,
        })
    }

    /// Create a weighted multinomial replacement sampler from a validated weight snapshot.
    pub fn weighted(weights: &[f64], seed: u64, epoch: u64) -> CacheResult<Self> {
        validate_weights(weights, weights.len())?;
        let distribution = WeightedIndex::new(weights)
            .map_err(|error| CacheError::InvalidInput(format!("invalid weights: {error}")))?;
        Ok(Self::Weighted {
            distribution,
            rng: epoch_rng(seed, epoch),
            remaining: weights.len(),
        })
    }

    /// Return the remaining epoch length for scheduler accounting.
    pub fn len(&self) -> usize {
        match self {
            Self::Uniform { remaining, .. } | Self::Weighted { remaining, .. } => *remaining,
        }
    }
}

impl Iterator for EpochSampler {
    type Item = usize;

    /// Draw the next physical sample index with replacement until the epoch is exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Uniform {
                rng,
                remaining,
                sample_count,
            } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                Some(rng.gen_range(0..*sample_count))
            }
            Self::Weighted {
                distribution,
                rng,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                Some(distribution.sample(rng))
            }
        }
    }
}

pub fn validate_weights(weights: &[f64], sample_count: usize) -> CacheResult<()> {
    if weights.len() != sample_count {
        return Err(CacheError::InvalidInput(format!(
            "expected {sample_count} weights, got {}",
            weights.len()
        )));
    }

    for (index, weight) in weights.iter().enumerate() {
        if !weight.is_finite() || *weight <= 0.0 {
            return Err(CacheError::InvalidInput(format!(
                "weight {index} must be positive and finite"
            )));
        }
    }

    Ok(())
}

/// Collect a weighted epoch for tests that need direct sequence comparison.
#[cfg(test)]
fn plan_epoch(weights: &[f64], seed: u64, epoch: u64) -> CacheResult<Vec<usize>> {
    Ok(EpochSampler::weighted(weights, seed, epoch)?.collect())
}

/// Derive the deterministic epoch RNG from the dataset seed and epoch counter.
fn epoch_rng(seed: u64, epoch: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ epoch.rotate_left(32))
}

#[cfg(test)]
mod tests {
    use super::plan_epoch;

    #[test]
    fn planned_epoch_is_deterministic() {
        let weights = vec![1.0, 2.0, 3.0];
        let first = plan_epoch(&weights, 42, 7);
        let second = plan_epoch(&weights, 42, 7);

        assert!(matches!((&first, &second), (Ok(left), Ok(right)) if left == right));
    }
}
