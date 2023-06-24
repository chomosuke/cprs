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

use io::*;

fn main() {
    let mut sc = Scanner::new(stdin());
    let mut pt = Printer::new(stdout());
    let line = sc.next_line();
    pt.print("You typed: ");
    pt.println(&line);
}
