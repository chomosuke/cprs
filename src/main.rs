#![allow(unused_imports, dead_code, clippy::needless_range_loop, unused_labels)]
use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    io::{stdin, stdout, BufReader},
    mem,
};

mod io;
mod binary_searchable;
mod multi_set;
mod graph;
mod indexed_vec;
mod math;
mod knapsack;
mod sort;

use io::*;

fn main() {
    demo_next_line()
}

fn demo_next_line() {
    let mut sc = Scanner::new(stdin());
    let mut pt = Printer::new(stdout());
    let line = sc.next_line();
    pt.print("You typed: ");
    pt.println(&line);
}

fn demo_next_i32() {
    let mut sc = Scanner::new(stdin());
    let mut pt = Printer::new(stdout());
    let x1 = sc.next::<i32>();
    let x2 = sc.next::<i32>();
    pt.print("You typed: ");
    pt.println(&x1);
    pt.println(&x2);
}
