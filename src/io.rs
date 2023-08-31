use std::collections::{HashSet, VecDeque};
use std::fmt::Display;
use std::io::{BufReader, BufWriter, Lines, Read, Write};
use std::marker::PhantomData;
use std::{any::type_name, io::BufRead, str::FromStr};

pub struct ScannerIter<'a, R: Read, T> {
    remaining: usize,
    sc: &'a mut Scanner<R>,
    item: PhantomData<T>,
}

impl<R: Read, T: FromStr> Iterator for ScannerIter<'_, R, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            Some(self.sc.next::<T>())
        }
    }
}

pub struct Scanner<R: Read> {
    tokens: VecDeque<String>,
    delimiters: Option<HashSet<char>>,
    lines: Lines<BufReader<R>>,
}
impl<R: Read> Scanner<R> {
    pub fn new(source: R) -> Self {
        Self {
            tokens: VecDeque::new(),
            delimiters: None,
            lines: BufReader::new(source).lines(),
        }
    }

    pub fn with_delimiters(source: R, delimiters: &[char]) -> Self {
        Self {
            tokens: VecDeque::new(),
            delimiters: Some(delimiters.iter().copied().collect()),
            lines: BufReader::new(source).lines(),
        }
    }

    pub fn next<T: FromStr>(&mut self) -> T {
        let token = loop {
            let front = self.tokens.pop_front();
            if let Some(token) = front {
                break token;
            }
            self.receive_input();
        };
        token.parse::<T>().unwrap_or_else(|_| {
            panic!(
                "Cannot convert input to type {}: {}",
                type_name::<T>(),
                token
            )
        })
    }

    pub fn next_n<T: FromStr>(&mut self, n: usize) -> ScannerIter<'_, R, T> {
        ScannerIter {
            remaining: n,
            sc: self,
            item: PhantomData,
        }
    }

    pub fn next_line(&mut self) -> String {
        assert!(self.tokens.is_empty(), "You have unprocessed token(s)");
        self.lines
            .next()
            .and_then(|e| e.ok())
            .expect("next_line failed.")
    }

    pub fn at_line_start(&mut self) -> bool {
        self.tokens.is_empty()
    }

    fn receive_input(&mut self) {
        let line = self
            .lines
            .next()
            .and_then(|e| e.ok())
            .expect("receive_input failed.");
        if let Some(delimiters) = &self.delimiters {
            for token in line.split(|c| delimiters.contains(&c)) {
                self.tokens.push_back(token.to_string());
            }
        } else {
            for token in line.split_whitespace() {
                self.tokens.push_back(token.to_string());
            }
        }
    }
}

/// Buffered printer, usually won't print out anything until right before the
/// program terminate. Not meant for interactive use.
pub struct Printer<W: Write> {
    writer: BufWriter<W>,
}
impl<W: Write> Printer<W> {
    pub fn new(destination: W) -> Self {
        Self {
            writer: BufWriter::new(destination),
        }
    }

    pub fn print(&mut self, s: impl Display) {
        self.writer
            .write_all(s.to_string().as_bytes())
            .expect("print failed.");
    }

    pub fn print_bytes(&mut self, b: &[u8]) {
        self.writer.write_all(b).expect("print_bytes failed.");
    }

    pub fn println(&mut self, s: impl Display) {
        self.print(s);
        self.newline();
    }

    pub fn newline(&mut self) {
        self.print_bytes(&[b'\n']);
    }

    pub fn print_iter(&mut self, mut iter: impl Iterator<Item = impl Display>) {
        if let Some(e) = iter.next() {
            self.print(&e);
            for e in iter {
                self.print_bytes(&[b' ']);
                self.print(&e);
            }
        }
        self.newline();
    }
}
impl<W: Write> Drop for Printer<W> {
    fn drop(&mut self) {
        self.writer
            .flush()
            .expect("flush failed when dropping Printer.");
    }
}
