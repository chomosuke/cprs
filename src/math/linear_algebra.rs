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
