use std::{collections::VecDeque, ptr};

pub trait MergeSort {
    fn merge_sort(&mut self);
}

impl<T> MergeSort for Vec<T>
where
    T: PartialOrd,
{
    fn merge_sort(&mut self) {
        let self_as_slice = self.as_mut_slice();
        self_as_slice.merge_sort();
    }
}

impl<T> MergeSort for [T]
where
    T: PartialOrd,
{
    #[inline]
    fn merge_sort(&mut self) {
        merge_sort(self)
    }
}

fn merge_sort<T>(slice: &mut [T])
where
    T: PartialOrd,
{
    let mut size = 1;
    while size < slice.len() {
        for i in 0..=slice.len() / size / 2 {
            let i1 = i * 2 * size;
            let i2 = (i * 2 + 1) * size;
            let i2 = i2.min(slice.len());
            let i3 = (i + 1) * 2 * size; // this is guarantteed to be smaller or equals to vec.len()
            let i3 = i3.min(slice.len());
            merge(&mut slice[i1..i3], i2 - i1);
        }
        size *= 2;
    }
}

fn merge<T>(slice: &mut [T], mid: usize)
where
    T: PartialOrd,
{
    let mut s = slice.as_mut_ptr();

    let mut v1 = unsafe { fill_into_vec(&slice, mid) };

    let mut s1 = v1.as_ptr();

    let (e1, mut s2, e2) = unsafe {
        (
            v1.as_ptr().add(v1.len()),
            slice.as_ptr().add(mid),
            slice.as_ptr().add(slice.len()),
        )
    };

    //SAFETY: this is always indexing into known size
    unsafe {
        while s1 != e1 && s2 != e2 {
            if *s1 < *s2 {
                // SAFETY: ptr::copy() does not drop the value being over written
                ptr::copy(s1, s, 1);
                s1 = s1.add(1);
            } else {
                ptr::copy(s2, s, 1);
                s2 = s2.add(1);
            }
            s = s.add(1);
        }
        if s1 != e1 {
            ptr::copy(s1, s, e1.offset_from(s1) as usize);
        }
        // SAFETY: set len to zero so that the content does not get dropped.
        v1.set_len(0);
        // SAFETY: nothing of value T gets dropped here and if my logic is correct nothing will be
        // leaked.
    }
}

unsafe fn fill_into_vec<T>(slice: &[T], size: usize) -> Vec<T>
where
    T: PartialOrd,
{
    let mut vec = Vec::<T>::with_capacity(size);
    ptr::copy(slice.as_ptr(), vec.as_mut_ptr(), size);
    vec.set_len(size);
    vec
}

#[test]
fn test_merge_sort_self_slice() {
    let mut vec: Vec<u16> = vec![
        1, 54, 618, 45, 32, 35, 6480, 573, 18, 816, 31, 21, 0, 789, 645, 321, 591, 64, 28,
    ];
    let mut vec_2 = vec.clone();
    let slice_1: &mut [u16] = vec.as_mut_slice();
    let slice_2: &mut [u16] = vec_2.as_mut_slice();

    slice_1.sort();
    slice_2.merge_sort();

    assert_eq!(slice_1, slice_2)
}

#[test]
fn test_merge_sort_self_vec() {
    let mut vec: Vec<u16> = vec![
        1, 54, 618, 45, 32, 35, 6480, 573, 18, 816, 31, 21, 0, 789, 645, 321, 591, 64, 28,
    ];
    let mut sorted = vec.clone();

    sorted.sort();
    vec.merge_sort();

    assert_eq!(sorted, vec)
}
