use std::ops::Index;

struct Rotatable<'a, T> {
    offset: usize,
    ts: &'a [T],
}

impl<'a, T> Rotatable<'a, T> {
    fn new(ts: &'a [T]) -> Self {
        Rotatable { offset: 0, ts }
    }

    fn rotate(&mut self, offset: usize) {
        self.offset += offset;
        self.offset %= self.ts.len();
    }
}

impl<'a, T> Index<usize> for Rotatable<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &'a Self::Output {
        &self.ts[(index + self.offset) % self.ts.len()]
    }
}

fn lexicographically_minimal_rotation<T>(ts: &[T]) -> Rotatable<'_, T>
where
    T: PartialOrd,
{
    let mut rotated = Rotatable::new(ts);
    if ts.is_empty() {
        return rotated;
    }

    let mut pre_suf = vec![0_usize; ts.len() * 2];
    let mut index = 1;
    while index < pre_suf.len() {
        let mut pre_len = pre_suf[index - 1];
        loop {
            if rotated[index] == rotated[pre_len] {
                pre_suf[index] = pre_len + 1;
                break;
            } else if rotated[index] < rotated[pre_len] {
                rotated.rotate(index - pre_len);
                if pre_len > 0 {
                    index = pre_len - 1;
                } else {
                    index = 0;
                }
                break;
            }
            if pre_len == 0 {
                pre_suf[index] = 0;
                break;
            }
            pre_len = pre_suf[pre_len - 1];
        }
        index += 1;
    }

    rotated
}

#[test]
fn test() {
    assert_eq!(lexicographically_minimal_rotation(&"dbca".bytes().collect::<Vec<_>>()).offset, 3);
    assert_eq!(lexicographically_minimal_rotation(&"abbab".bytes().collect::<Vec<_>>()).offset, 3);
    assert_eq!(lexicographically_minimal_rotation(&"aabaaa".bytes().collect::<Vec<_>>()).offset, 3);
    assert_eq!(lexicographically_minimal_rotation(&"".bytes().collect::<Vec<_>>()).offset, 0);
    assert_eq!(
        lexicographically_minimal_rotation(&"ababbababababbababb".bytes().collect::<Vec<_>>()).offset,
        5
    );
    assert_eq!(
        lexicographically_minimal_rotation(&"1111111101".bytes().collect::<Vec<_>>()).offset,
        8
    );
}
