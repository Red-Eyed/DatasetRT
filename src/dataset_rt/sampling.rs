use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::types::{CacheError, CacheResult};

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

pub fn plan_epoch(weights: &[f64], seed: u64, epoch: u64) -> CacheResult<Vec<usize>> {
    validate_weights(weights, weights.len())?;
    let distribution = WeightedIndex::new(weights)
        .map_err(|error| CacheError::InvalidInput(format!("invalid weights: {error}")))?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ epoch.rotate_left(32));
    let epoch_len = weights.len();
    Ok((0..epoch_len)
        .map(|_| distribution.sample(&mut rng))
        .collect())
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
