use std::mem;

type Int = i128;

fn get_gcd(mut a: Int, mut b: Int) -> Int {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

fn find_linear_comb_2(a: Int, b: Int) -> (Int, Int, Int) {
    if b == 0 {
        return (a, 1, 0);
    }
    let rem = a % b;
    let (gcd, x1, y1) = find_linear_comb_2(b, rem);
    let x2 = y1;
    let y2 = x1 - a / b * y1;
    (gcd, x2, y2)
}

fn find_linear_comb(a: Int, b: Int) -> (Int, Int, Int) {
    let mut x_a1 = 1;
    let mut y_a1 = 0;
    let mut x_b1 = 0;
    let mut y_b1 = 1;
    let mut a1 = a;
    let mut b1 = b;
    while b1 != 0 {
        let q = a1 / b1;
        let a2 = b1;
        let b2 = a1 - q * b1;
        let x_a2 = x_b1;
        let y_a2 = y_b1;
        let x_b2 = x_a1 - q * x_b1;
        let y_b2 = y_a1 - q * y_b1;

        a1 = a2;
        b1 = b2;
        x_a1 = x_a2;
        y_a1 = y_a2;
        x_b1 = x_b2;
        y_b1 = y_b2;
    }
    (a1, x_a1, y_a1)
}

#[test]
fn test() {
    assert_eq!(get_gcd(12, 18), 6);

    let (a, b) = (79, 33);
    let (gcd, x, y) = find_linear_comb_2(a, b);
    assert_eq!(x * a + y * b, gcd);
    assert_eq!(gcd, get_gcd(a, b));

    let (gcd2, x2, y2) = find_linear_comb(a, b);
    assert_eq!((gcd, x, y), (gcd2, x2, y2));
}
