mod fibonacci;
mod permutate;
mod pow;

fn apply_n<E: Clone>(x: &E, n: usize, f: &impl Fn(&E, &E) -> E) -> E {
    if n == 0 {
        panic!("This function does not have an Id element.");
    } else if n == 1 {
        x.clone()
    } else if n % 2 == 0 {
        if n / 2 == 1 {
            f(x, x)
        } else {
            let x = apply_n(x, n / 2, f);
            f(&x, &x)
        }
    } else {
        f(&apply_n(x, n - 1, f), x)
    }
}

#[test]
fn test() {
    assert_eq!(30, apply_n(&1, 30, &|a, b| { a + b }));
}
