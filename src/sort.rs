use std::{collections::VecDeque, ptr};

pub fn merge_sort<T>(mut vec: Vec<T>) -> Vec<T>
where
    T: PartialOrd,
{
    let mut size = 1;
    while size < vec.len() {
        for i in 0..=vec.len() / size / 2 {
            let i1 = i * 2 * size;
            let i2 = (i * 2 + 1) * size;
            let i2 = i2.min(vec.len());
            let i3 = (i + 1) * 2 * size; // this is guarantteed to be smaller or equals to vec.len()
            let i3 = i3.min(vec.len());
            merge(&mut vec[i1..i3], i2 - i1);
        }
        size *= 2;
    }
    vec
}

fn merge<T: PartialOrd>(slice: &mut [T], mid: usize) {
    unsafe {
        let mut s = slice.as_mut_ptr();

        let mut v1 = Vec::<T>::with_capacity(mid);
        ptr::copy(slice.as_ptr(), v1.as_mut_ptr(), mid);
        v1.set_len(mid);
        let mut s1 = v1.as_ptr();
        let e1 = v1.as_ptr().add(v1.len());

        let mut s2 = slice.as_ptr().add(mid);
        let e2 = slice.as_ptr().add(slice.len());

        while s1 != e1 && s2 != e2 {
            if *s1 < *s2 {
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
        v1.set_len(0);
    }
}

#[test]
fn test_merge_sort() {
    let vec: Vec<u32> = vec![
        1, 54, 618, 45, 32, 35, 6480, 573, 18, 816, 31, 21, 0, 789, 645, 321, 591, 64, 28,
    ];
    let mut sorted = vec.clone();
    sorted.sort();
    let vec = merge_sort(vec);
    assert_eq!(sorted, vec);
}
