use std::cmp::max;
use std::collections::HashMap;

type Value = u64;
type Weight = u64;

pub fn knapsack_01(items: &[(Value, Weight)], max_w: u64) -> Value {
    // BFS but prune weight > max_w
    let mut weight_maxv = HashMap::new();
    weight_maxv.insert(0, 0);
    for &(v, w) in items {
        let mut next_weight_maxv = HashMap::new();
        for (weight, maxv) in weight_maxv {
            // exclude
            let next_maxv = next_weight_maxv.entry(weight).or_default();
            *next_maxv = max(*next_maxv, maxv);
            if weight + w <= max_w {
                // include
                let next_maxv = next_weight_maxv.entry(weight + w).or_default();
                *next_maxv = max(*next_maxv, maxv + v);
            }
        }
        weight_maxv = next_weight_maxv;
    }
    *weight_maxv.values().max().unwrap()
}

#[test]
fn test_knapsack_01() {
    let r = knapsack_01(&[(10, 2), (20, 4), (30, 6)], 8);
    assert_eq!(40, r);
}