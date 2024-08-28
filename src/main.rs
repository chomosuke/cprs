#![allow(
    unused_imports,
    dead_code,
    clippy::needless_range_loop,
    unused_labels,
    clippy::ptr_arg
)]
use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    io::{stdin, stdout, BufReader},
    mem,
};

mod algebra;
mod bag;
mod binary_search;
mod bit;
mod io;
mod knapsack;
mod multi_set;
mod segment_tree;
mod sort;
mod string;
mod tree;

use io::*;

fn main() {
    demo_next_i32()
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
    if !sc.at_line_start() {
        pt.println("Unexpected input. Expected format: <i32> <i32>");
        return;
    }

    pt.println("You typed: ");
    pt.println(x1);
    pt.println(x2);
}
