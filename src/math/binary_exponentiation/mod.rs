mod fibonacci;
mod permutate;
mod pow;

fn apply_n<I: Copy>(x: I, n: usize, f: &impl Fn(I, I) -> I) -> I {
    if n == 0 {
        panic!("This function does not have an Id element.");
    } else if n == 1 {
        x
    } else if n % 2 == 0 {
        let x = apply_n(x, n / 2, f);
        f(x, x)
    } else {
        f(apply_n(x, n - 1, f), x)
    }
}

#[test]
fn test() {
    assert_eq!(30, apply_n(1, 30, &|a, b| { a + b }));
}
