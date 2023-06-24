use core::hash::Hash;
use std::collections::{hash_map, HashMap};

pub struct MultiSetIter<'a, E, I> {
    elem_count: Option<(&'a E, usize)>,
    count_iter: I,
}
impl<'a, E, I: Iterator<Item = (&'a E, &'a usize)>> MultiSetIter<'a, E, I> {
    fn new(count_iter: I) -> Self {
        Self {
            elem_count: None,
            count_iter,
        }
    }
}
impl<'a, E, I: Iterator<Item = (&'a E, &'a usize)>> Iterator for MultiSetIter<'a, E, I> {
    type Item = &'a E;

    /// O(capacity)
    fn next(&mut self) -> Option<Self::Item> {
        while self.elem_count.is_none() || self.elem_count.unwrap().1 == 0 {
            if let Some((e, &c)) = self.count_iter.next() {
                self.elem_count = Some((e, c));
            } else {
                return None;
            }
        }
        self.elem_count.as_mut().unwrap().1 -= 1;
        Some(self.elem_count.unwrap().0)
    }
}

#[derive(Clone)]
pub struct MultiSet<E> {
    count_map: HashMap<E, usize>,
}
impl<E: Eq + Hash> MultiSet<E> {
    pub fn new() -> Self {
        Self {
            count_map: HashMap::new(),
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            count_map: HashMap::with_capacity(capacity),
        }
    }

    /// O(1)
    pub fn count(&self, e: &E) -> usize {
        *self.count_map.get(e).unwrap_or(&0)
    }

    /// O(1)
    pub fn insert(&mut self, e: E) {
        let next = self.count(&e) + 1;
        self.count_map.insert(e, next);
    }

    /// O(1)
    pub fn remove<'a>(&mut self, e: &'a E) -> Option<&'a E> {
        let next = self.count(e) as i128 - 1;
        if next < 0 {
            None
        } else {
            *self.count_map.get_mut(e).unwrap() = next as usize;
            Some(e)
        }
    }

    pub fn iter(&self) -> MultiSetIter<'_, E, hash_map::Iter<'_, E, usize>> {
        MultiSetIter::new(self.count_map.iter())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use std::iter::FromIterator;

    use super::*;

    #[test]
    fn multi_set() {
        let mut s = MultiSet::new();
        s.insert(1);
        s.insert(2);
        s.insert(1);
        assert_eq!(s.count(&1), 2);
        assert_eq!(s.count(&2), 1);
        assert_eq!(s.count(&3), 0);
        assert_eq!(
            s.iter().collect::<HashSet<_>>(),
            HashSet::from_iter(&[1, 1, 2])
        );
        s.remove(&2);
        s.remove(&1);
        assert_eq!(s.count(&1), 1);
        assert_eq!(s.count(&2), 0);
    }
}
