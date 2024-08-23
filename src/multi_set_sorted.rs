use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    mem,
};

enum MultiSetE<'a, E, F> {
    Real { inner: E, cmp: &'a F },
    Fake { inner: &'a E, cmp: &'a F },
}

impl<'a, E, F> PartialEq for MultiSetE<'a, E, F>
where
    F: Fn(&E, &E) -> Ordering + 'a,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<'a, E, F> PartialOrd for MultiSetE<'a, E, F>
where
    F: Fn(&E, &E) -> Ordering + 'a,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a, E, F> Eq for MultiSetE<'a, E, F> where F: Fn(&E, &E) -> Ordering + 'a {}
impl<'a, E, F> Ord for MultiSetE<'a, E, F>
where
    F: Fn(&E, &E) -> Ordering + 'a,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            Self::Real { inner, cmp } => match other {
                Self::Real { inner: inner2, .. } => cmp(inner, inner2),
                Self::Fake { inner: inner2, .. } => cmp(inner, inner2),
            },
            Self::Fake { inner, cmp } => match other {
                Self::Real { inner: inner2, .. } => cmp(inner, inner2),
                Self::Fake { inner: inner2, .. } => cmp(inner, inner2),
            },
        }
    }
}

struct SortedMultiSet<'a, E, F> {
    inner: BTreeMap<MultiSetE<'a, E, F>, BTreeSet<E>>,
    cmp: F,
}

impl<'a, E, F> SortedMultiSet<'a, E, F>
where
    F: Fn(&E, &E) -> Ordering + 'a,
    E: Ord,
{
    fn new(ord: F) -> Self {
        Self {
            inner: BTreeMap::new(),
            cmp: ord,
        }
    }

    fn insert(&'a mut self, e: E) {
        let we = MultiSetE::Real {
            inner: e,
            cmp: &self.cmp,
        };
        if let Some(inner_set) = self.inner.get_mut(&we) {
            let MultiSetE::Real { inner: e, .. } = we else {
                unreachable!()
            };
            inner_set.insert(e);
        } else {
            self.inner.insert(we, BTreeSet::new());
        }
    }

    fn remove_all<'b>(&'a mut self, e: &'b E) -> Option<BTreeSet<E>> {
        let e = MultiSetE::Fake {
            inner: unsafe { mem::transmute(e) },
            cmp: &self.cmp,
        };
        let (MultiSetE::Real { inner, .. }, mut tree) = self.inner.remove_entry(&e)? else {
            unreachable!()
        };
        tree.insert(inner);
        Some(tree)
    }
}
