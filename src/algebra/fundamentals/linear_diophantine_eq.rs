use std::{cmp::Ordering, ops::Range};

type Int = i128;

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

struct LDESIter {
    xy: (Int, Int),
    diff: (Int, Int),
    k_range: Range<Int>,
}
impl Iterator for LDESIter {
    type Item = (Int, Int);

    fn next(&mut self) -> Option<Self::Item> {
        let k = self.k_range.next()?;
        Some((self.xy.0 + k * self.diff.0, self.xy.1 + k * self.diff.1))
    }
}

fn linear_diophantine_eq_sols(
    a: Int,
    b: Int,
    c: Int,
    x_range: Range<Int>,
    y_range: Range<Int>,
) -> impl Iterator<Item = (Int, Int)> {
    let (gcd, x, y) = find_linear_comb(a, b);
    if c % gcd != 0 {
        return LDESIter {
            xy: (0, 0),
            diff: (0, 0),
            k_range: 0..0,
        };
    }
    let n = c / gcd;
    let x = x * n;
    let y = y * n;

    let mut x_diff = b / gcd;
    let mut y_diff = -a / gcd;
    if x_diff < 0 {
        x_diff = -x_diff;
        y_diff = -y_diff;
    }

    let x_end = x_range.end - 1;
    let x_start = x_range.start;
    let y_end = y_range.end - 1;
    let y_start = y_range.start;

    fn div_ceil(x: Int, y: Int) -> Int {
        if (x < 0) != (y < 0) {
            x / y
        } else {
            (x - 1) / y + 1
        }
    }

    fn div_floor(x: Int, y: Int) -> Int {
        if (x < 0) != (y < 0) {
            (x + 1) / y - 1
        } else {
            x / y
        }
    }

    let mut k_start = Int::MIN;
    let mut k_end = Int::MAX;
    if x_diff == 0 {
        if x_start <= x && x < x_end {
            return LDESIter {
                xy: (x, 0),
                diff: (0, 1),
                k_range: y_start..(y_end + 1),
            };
        } else {
            return LDESIter {
                xy: (0, 0),
                diff: (0, 0),
                k_range: 0..0,
            };
        }
    }
    if x_diff < 0 {
        k_start = k_start.max(div_ceil(x_end - x, x_diff));
        k_end = k_end.min(div_floor(x_start - x, x_diff));
    } else {
        k_start = k_start.max(div_ceil(x_start - x, x_diff));
        k_end = k_end.min(div_floor(x_end - x, x_diff));
    }

    if y_diff == 0 {
        if y_start <= y && y < y_end {
            return LDESIter {
                xy: (0, y),
                diff: (1, 0),
                k_range: x_start..(x_end + 1),
            };
        } else {
            return LDESIter {
                xy: (0, 0),
                diff: (0, 0),
                k_range: 0..0,
            };
        }
    }
    if y_diff < 0 {
        k_start = k_start.max(div_ceil(y_end - y, y_diff));
        k_end = k_end.min(div_floor(y_start - y, y_diff));
    } else {
        k_start = k_start.max(div_ceil(y_start - y, y_diff));
        k_end = k_end.min(div_floor(y_end - y, y_diff));
    }

    LDESIter {
        xy: (x, y),
        diff: (x_diff, y_diff),
        k_range: k_start..(k_end + 1),
    }
}

fn linear_diophantine_eq_sols_bf(
    a: Int,
    b: Int,
    c: Int,
    x_range: Range<Int>,
    y_range: Range<Int>,
) -> Vec<(Int, Int)> {
    let mut sols = Vec::new();

    for x in x_range {
        for y in y_range.clone() {
            if x * a + y * b == c {
                sols.push((x, y));
            }
        }
    }
    sols
}

#[test]
fn test() {
    let abcs = [
        (77, 33, 22),
        (77, -33, -22),
        (-77, 33, -55),
        (-77, -33, 55),
        (-77, -33, -55),
    ];
    for (a, b, c) in abcs {
        let sols = linear_diophantine_eq_sols(a, b, c, -50..50, -50..50).collect::<Vec<_>>();
        let sols2 = linear_diophantine_eq_sols_bf(a, b, c, -50..50, -50..50);
        assert_eq!(sols, sols2, "a: {a}, b: {b}, c: {c}");
    }
}
