type I = i128;

fn pow(x: I, n: I, m: I) -> I {
    let x = x.rem_euclid(m);
    if n == 0 {
        1
    } else if n % 2 == 0 {
        pow(x, n / 2, m).pow(2).rem_euclid(m)
    } else {
        (pow(x, n - 1, m) * x).rem_euclid(m)
    }
}
