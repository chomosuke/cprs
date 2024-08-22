type Int = u128;

fn mat_multi(m1: &Vec<Vec<Int>>, m2: &Vec<Vec<Int>>) -> Vec<Vec<Int>> {
    let mut mr = vec![vec![0; m2[0].len()]; m1.len()];
    assert_eq!(m1[0].len(), m2.len());
    for i in 0..m1.len() {
        for j in 0..m2[0].len() {
            for k in 0..m2.len() {
                mr[i][j] += m2[k][j] * m1[i][k];
            }
        }
    }
    mr
}

fn mat_pow(m: &Vec<Vec<Int>>, n: Int) -> Vec<Vec<Int>> {
    if n == 0 {
        let mut mr = vec![vec![0; m.len()]; m.len()];
        for i in 0..m.len() {
            mr[i][i] = 1;
        }
        mr
    } else if n == 1 {
        m.clone()
    } else if n % 2 == 0 {
        let m2 = mat_pow(m, n / 2);
        mat_multi(&m2, &m2)
    } else {
        mat_multi(m, &mat_pow(m, n - 1))
    }
}

fn fibonacci(n: Int) -> Int {
    let v = vec![vec![0], vec![1]];
    let f = vec![vec![0, 1], vec![1, 1]];
    let f_n = mat_pow(&f, n);
    let v = mat_multi(&f_n, &v);
    v[0][0]
}

#[test]
fn test_fibonacci() {
    let fibs = [
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
    ];
    for (i, f) in fibs.into_iter().enumerate() {
        assert_eq!(f, fibonacci(i as Int), "{i}");
    }
}
