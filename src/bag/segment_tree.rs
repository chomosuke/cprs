use std::ops::Bound::*;
use std::ops::{Deref, RangeBounds};

const USIZE_BITS: u32 = 64;
fn log2_ceil(x: usize) -> u32 {
    if x == 0 {
        0
    } else {
        USIZE_BITS - (x - 1).leading_zeros()
    }
}
fn pow2_ceil(x: usize) -> usize {
    let n = log2_ceil(x);
    2usize.pow(n)
}

pub struct SegmentTree<E, F> {
    combine: F,
    inner: Vec<E>,
    tree: Vec<E>,
    inner_cap: usize,
    zero: E,
}
impl<E, F> Deref for SegmentTree<E, F> {
    type Target = Vec<E>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<E: Clone, F: Fn(&E, &E) -> E> SegmentTree<E, F> {
    fn parent(i: usize) -> usize {
        i / 2
    }
    fn left(i: usize) -> usize {
        2 * i
    }
    fn right(i: usize) -> usize {
        2 * i + 1
    }

    pub fn new(zero: E, combine: F) -> Self {
        Self {
            combine,
            inner: Vec::new(),
            tree: Vec::new(),
            inner_cap: 0,
            zero,
        }
    }
    pub fn with_capacity(capacity: usize, zero: E, combine: F) -> Self {
        Self {
            combine,
            inner: Vec::with_capacity(capacity),
            tree: Vec::with_capacity(pow2_ceil(capacity) * 2),
            inner_cap: 0,
            zero,
        }
    }
    /// O(n)
    pub fn from_vec(vec: Vec<E>, zero: E, combine: F) -> Self {
        let mut iv = Self {
            combine,
            inner: vec,
            zero,
            tree: Vec::new(),
            inner_cap: 0,
        };
        iv.rebuild();
        iv
    }

    /// O(n)
    fn rebuild(&mut self) {
        let inner = &mut self.inner;
        let combine = &self.combine;
        let zero = &self.zero;
        let inner_cap = pow2_ceil(inner.len());
        let mut tree = vec![zero.clone(); inner_cap * 2];
        tree[inner_cap..(inner_cap + inner.len())].clone_from_slice(&inner[..]);
        let mut n = inner_cap;
        while n > 1 {
            n /= 2;
            for i in n..(n * 2) {
                tree[i] = combine(&tree[Self::left(i)], &tree[Self::right(i)]);
            }
        }
        self.tree = tree;
        self.inner_cap = inner_cap;
    }

    /// O(log(n))
    pub fn query(&self, rng: impl RangeBounds<usize>) -> E {
        let start = match rng.start_bound() {
            Excluded(x) => x + 1,
            Included(x) => *x,
            Unbounded => 0,
        };
        let end = match rng.end_bound() {
            Excluded(x) => x - 1,
            Included(x) => *x,
            Unbounded => self.inner.len() - 1,
        };
        let mut start = start + self.inner_cap;
        let mut end = end + self.inner_cap;
        let mut result = self.zero.clone();
        while start <= end {
            if start % 2 == 1 {
                result = (self.combine)(&result, &self.tree[start]);
                start += 1;
            }
            if end % 2 == 0 {
                result = (self.combine)(&result, &self.tree[end]);
                end -= 1;
            }
            start = Self::parent(start);
            end = Self::parent(end);
        }
        result
    }

    fn update(&mut self, index: usize) {
        self.tree[index + self.inner_cap] = if index < self.inner.len() {
            self.inner[index].clone()
        } else {
            self.zero.clone()
        };
        let mut index = index + self.inner_cap;
        while index > 1 {
            index = Self::parent(index);
            self.tree[index] = (self.combine)(
                &self.tree[Self::left(index)],
                &self.tree[Self::right(index)],
            );
        }
    }
    /// O(log(n))
    pub fn push(&mut self, e: E) {
        self.inner.push(e);
        if self.inner.len() > self.inner_cap {
            self.rebuild();
        } else {
            self.update(self.inner.len() - 1);
        }
    }
    /// O(log(n))
    pub fn pop(&mut self) -> Option<E> {
        let e = self.inner.pop();
        self.update(self.inner.len());
        e
    }
    /// O(log(n))
    pub fn set(&mut self, i: usize, e: E) {
        self.inner[i] = e;
        self.update(i);
    }
}

#[cfg(test)]
mod test {
    use super::SegmentTree;

    #[test]
    fn test() {
        let mut iv = SegmentTree::from_vec(vec![1, 3, 4, 8, 6, 1, 4, 2], std::i32::MAX, |a, b| {
            if a < b {
                *a
            } else {
                *b
            }
        });
        assert_eq!(iv.query(1..7), 1);
        iv.set(5, 100);
        assert_eq!(iv.query(1..7), 3);
        iv.push(-2);
        assert_eq!(iv.query(..), -2);
        iv.set(8, 100);
        assert_eq!(iv.query(7..=8), 2);
        iv.set(8, -2);
        assert_eq!(iv.query(7..=8), -2);
        assert_eq!(iv.pop(), Some(-2));
        assert_eq!(iv.query(..), 1);
    }
}
