fn common_prefix_suffix_len<T: Eq>(s: &[T]) -> Vec<usize> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut f = Vec::with_capacity(s.len());
    f.push(0);
    for i in 1..s.len() {
        let mut j = f[i - 1];
        loop {
            if s[i] == s[j] {
                f.push(j + 1);
                break;
            }
            if j == 0 {
                f.push(0);
                break;
            }
            j = f[j - 1];
        }
    }
    f
}

#[test]
fn test() {
    assert_eq!(
        common_prefix_suffix_len("dbca".as_bytes()),
        vec![0, 0, 0, 0],
    );
    assert_eq!(
        common_prefix_suffix_len("abbab".as_bytes()),
        vec![0, 0, 0, 1, 2],
    );
    assert_eq!(
        common_prefix_suffix_len("aabaaa".as_bytes()),
        vec![0, 1, 0, 1, 2, 2],
    );
    assert_eq!(
        common_prefix_suffix_len("ababbababababbababb".as_bytes()),
        vec![0, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4, 3, 4, 5, 6, 7, 8, 9, 5],
    );
}
