type U = u64;

/// Will look for first i such that p(i) == false.
fn search(start: U, step: U, p: impl Fn(U) -> bool) -> U {
    assert!(p(start));
    let mut index = start;
    let mut step = step;
    while step > 0 {
        if p(index + step) {
            index += step;
        } else {
            step /= 2;
        }
    }
    index + 1
}
