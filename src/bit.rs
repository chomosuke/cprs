type N = u128;

fn get_nth_bit(a: N, n: usize) -> bool {
    (a >> n) % 2 == 1
}

fn set_nth_bit(a: &mut N, n: usize) {
    *a = *a & (1 << n)
}