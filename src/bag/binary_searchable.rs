use std::cmp::*;

pub trait BinarySearchable<T> {
    fn binary_search_leq(&self, x: &T) -> usize;
    fn binary_search_geq(&self, x: &T) -> usize;
}
impl<T: Ord> BinarySearchable<T> for [T] {
    fn binary_search_leq(&self, x: &T) -> usize {
        self.binary_search_by(|p| {
            let r = p.cmp(x);
            if r == Ordering::Greater {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        })
        .err()
        .unwrap()
    }

    fn binary_search_geq(&self, x: &T) -> usize {
        self.binary_search_by(|p| {
            let r = p.cmp(x);
            if r == Ordering::Less {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .err()
        .unwrap()
    }
}
