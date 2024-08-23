fn apply_perm(perm: &[usize], arr: &[usize]) -> Vec<usize> {
    assert_eq!(perm.len(), arr.len());
    let mut res = Vec::with_capacity(arr.len());
    for &p in perm {
        res.push(arr[p]);
    }
    res
}

/// 0 indexed
fn permutate(perm: &[usize], times: usize) -> Vec<usize> {
    if times == 0 {
        (0..perm.len()).collect()
    } else if times % 2 == 0 {
        let p = permutate(perm, times / 2);
        apply_perm(&p, &p)
    } else {
        let p = permutate(perm, times - 1);
        apply_perm(perm, &p)
    }
}

#[test]
fn test() {
    let arr = (0..10).collect::<Vec<_>>();
    let perm = [0, 8, 7, 3, 1, 5, 9, 6, 4, 2];
    assert_eq!(apply_perm(&perm, &arr), perm);
    let mut res_perm = arr.clone();
    for _ in 0..17 {
        res_perm = apply_perm(&perm, &res_perm);
    }
    assert_eq!(res_perm, apply_perm(&permutate(&perm, 17), &arr))
}
