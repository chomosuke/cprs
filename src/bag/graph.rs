const USIZE_BITS: u32 = 64;
fn log2_ceil(x: usize) -> u32 {
    if x == 0 {
        0
    } else {
        USIZE_BITS - (x - 1).leading_zeros()
    }
}
fn pow2_ceil(x: usize) -> usize {
    let n = log2_ceil(x);
    2usize.pow(n)
}
fn highest_one_bit(x: usize) -> u32 {
    USIZE_BITS - x.leading_zeros()
}

use std::collections::VecDeque;

pub struct DepthFirstIter<'a> {
    adj_nodess: &'a Vec<Vec<(usize, i128)>>,
    pushed: Vec<bool>,
    stack: VecDeque<usize>,
}
impl<'a> DepthFirstIter<'a> {
    pub fn new(adj_nodess: &'a Vec<Vec<(usize, i128)>>, start: usize) -> Self {
        let capacity = adj_nodess.len();
        let mut pushed = vec![false; capacity];
        let mut stack = VecDeque::with_capacity(capacity);
        stack.push_back(start);
        pushed[start] = true;
        Self {
            adj_nodess,
            pushed,
            stack,
        }
    }
}
impl Iterator for DepthFirstIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let DepthFirstIter {
            adj_nodess,
            pushed,
            stack,
        } = self;
        if let Some(node) = stack.pop_back() {
            for &(node, _) in adj_nodess[node].iter().rev() {
                if !pushed[node] {
                    stack.push_back(node);
                    pushed[node] = true;
                }
            }
            Some(node)
        } else {
            None
        }
    }
}
pub struct BreathFirstIter<'a> {
    adj_nodess: &'a Vec<Vec<(usize, i128)>>,
    pushed: Vec<bool>,
    queue: VecDeque<usize>,
}
impl<'a> BreathFirstIter<'a> {
    pub fn new(adj_nodess: &'a Vec<Vec<(usize, i128)>>, start: usize) -> Self {
        let capacity = adj_nodess.len();
        let mut pushed = vec![false; capacity];
        let mut queue = VecDeque::with_capacity(capacity);
        queue.push_back(start);
        pushed[start] = true;
        Self {
            adj_nodess,
            pushed,
            queue,
        }
    }
}
impl Iterator for BreathFirstIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let BreathFirstIter {
            adj_nodess,
            pushed,
            queue,
        } = self;
        if let Some(node) = queue.pop_front() {
            for &(node, _) in adj_nodess[node].iter() {
                if !pushed[node] {
                    queue.push_back(node);
                    pushed[node] = true;
                }
            }
            Some(node)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test_search_graph {
    use super::*;

    const EDGES: &[(usize, usize, i128)] = &[
        (0, 1, 5),
        (0, 2, 3),
        (0, 3, 7),
        (1, 0, 5),
        (1, 3, 3),
        (1, 4, 2),
        (2, 0, 3),
        (2, 3, 1),
        (3, 0, 7),
        (3, 1, 3),
        (3, 2, 1),
        (3, 4, 2),
        (4, 1, 2),
        (4, 3, 2),
    ];

    #[test]
    fn dfs() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(
            DepthFirstIter::new(g.get_adj_nodess(), 0).collect::<Vec<_>>(),
            vec![0, 1, 4, 2, 3]
        );
    }

    #[test]
    fn bfs() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(
            BreathFirstIter::new(g.get_adj_nodess(), 0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }
}

use std::iter::FromIterator;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
};

#[derive(Clone)]
pub struct MultiSet<E> {
    count_map: HashMap<E, usize>,
}
impl<E: Eq + Hash> MultiSet<E> {
    pub fn new() -> Self {
        Self {
            count_map: HashMap::new(),
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            count_map: HashMap::with_capacity(capacity),
        }
    }

    /// O(1)
    pub fn count(&self, e: &E) -> usize {
        *self.count_map.get(e).unwrap_or(&0)
    }

    /// O(1)
    pub fn insert(&mut self, e: E) {
        let next = self.count(&e) + 1;
        self.count_map.insert(e, next);
    }

    /// O(1)
    pub fn remove<'a>(&mut self, e: &'a E) -> Option<&'a E> {
        let next = self.count(e) as i128 - 1;
        if next < 0 {
            None
        } else {
            *self.count_map.get_mut(e).unwrap() = next as usize;
            Some(e)
        }
    }
}

/// assuming connected
/// average: O(m + nlog(m))
/// worst: O((m + n)log(m))
fn dijkstra<F: Fn(usize) -> bool>(
    adj_nodess: &[Vec<(usize, i128)>],
    start: usize,
    stop_when: F,
) -> Vec<i128> {
    let mut shortest_path_len = vec![std::i128::MAX; adj_nodess.len()];
    let mut queue = BinaryHeap::new();
    queue.push(Reverse((0, start)));
    while let Some(Reverse((distance, node))) = queue.pop() {
        if shortest_path_len[node] != std::i128::MAX {
            continue;
        }
        shortest_path_len[node] = distance;
        if stop_when(node) {
            break;
        }
        for &(adj_node, weight) in &adj_nodess[node] {
            queue.push(Reverse((distance + weight, adj_node)));
        }
    }
    shortest_path_len
}
/// worst: O(mn)
fn spfa(adj_nodess: &[Vec<(usize, i128)>], start: usize) -> Option<Vec<i128>> {
    let mut shortest_path_len = vec![std::i128::MAX; adj_nodess.len()];
    let mut shortest_path_edge_len = vec![0; adj_nodess.len()];
    shortest_path_len[start] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        for &(adj_node, weight) in &adj_nodess[node] {
            if shortest_path_len[node] + weight < shortest_path_len[adj_node] {
                shortest_path_len[adj_node] = shortest_path_len[node] + weight;
                shortest_path_edge_len[adj_node] = shortest_path_edge_len[node] + 1;
                if shortest_path_edge_len[adj_node] >= adj_nodess.len() {
                    // negative cycle
                    return None;
                }
                queue.push_back(adj_node);
            }
        }
    }
    Some(shortest_path_len)
}
/// O(n^3)
fn floyd_warshall(adj_nodess: &[Vec<(usize, i128)>]) -> Option<Vec<Vec<i128>>> {
    let n = adj_nodess.len();
    let mut shortest_path_lens = vec![vec![std::i128::MAX; n]; n];
    for (node, adj_nodes) in adj_nodess.iter().enumerate() {
        for &(adj_node, weight) in adj_nodes {
            shortest_path_lens[node][adj_node] = weight;
        }
    }
    for node in 0..n {
        shortest_path_lens[node][node] = 0;
    }
    for nodei in 0..n {
        for node1 in 0..n {
            for node2 in 0..n {
                if shortest_path_lens[node1][nodei] != std::i128::MAX
                    && shortest_path_lens[nodei][node2] != std::i128::MAX
                {
                    shortest_path_lens[node1][node2] = shortest_path_lens[node1][node2]
                        .min(shortest_path_lens[node1][nodei] + shortest_path_lens[nodei][node2]);
                }
            }
        }
    }
    for node in 0..n {
        if shortest_path_lens[node][node] < 0 {
            return None;
        }
    }
    Some(shortest_path_lens)
}

/// O(n + m)
fn reconstruct_shortest_path(
    rev_adj_nodess: &[Vec<(usize, i128)>],
    shortest_path_lens: &[i128],
    start: usize,
    end: usize,
) -> Option<Vec<usize>> {
    let dists_to_start = shortest_path_lens;
    assert!(
        dists_to_start[start] == 0,
        "expected shortest_path_lens[start] to be zero, looks like you got the wrong start node",
    );
    if dists_to_start[end] == std::i128::MAX {
        return None;
    }
    let mut shortest_path = Vec::new();
    let mut node = end;
    while node != start {
        shortest_path.push(node);
        node = rev_adj_nodess[node]
            .iter()
            .filter(|next_node| dists_to_start[next_node.0] == dists_to_start[node] - next_node.1)
            .min_by_key(|next_node| dists_to_start[next_node.0])
            .expect("shortest_path_lens corruption")
            .0;
    }
    shortest_path.push(node);
    Some(shortest_path.into_iter().rev().collect())
}
/// O(n + m)
fn reconstruct_all_shortest_path(
    adj_nodess: &Vec<Vec<(usize, i128)>>,
    shortest_path_lens: &[i128],
    start: usize,
) -> DirectedGraph {
    let dists_to_start = shortest_path_lens;
    assert!(
        dists_to_start[start] == 0,
        "expected shortest_path_lens[start] to be zero, looks like you got the wrong start node",
    );
    let mut edges = Vec::new();
    let mut visited = vec![false; adj_nodess.len()];
    fn dfs(
        node: usize,
        adj_nodess: &[Vec<(usize, i128)>],
        dists_to_start: &[i128],
        edges: &mut Vec<(usize, usize, i128)>,
        visited: &mut [bool],
    ) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        for &(adj_node, weight) in &adj_nodess[node] {
            if dists_to_start[adj_node] == dists_to_start[node] + weight {
                // some shortest path go through this edge
                edges.push((node, adj_node, weight));
                dfs(adj_node, adj_nodess, dists_to_start, edges, visited);
            }
        }
    }
    dfs(start, adj_nodess, dists_to_start, &mut edges, &mut visited);
    DirectedGraph::from_edges(edges, adj_nodess.len())
}

/// O(2^n * n^2)
fn hamiltonian_path(adj_nodess: &[Vec<(usize, i128)>]) -> Option<Vec<usize>> {
    let visited = vec![false; adj_nodess.len()];
    let mut memoize = HashMap::new();
    fn path(
        param: &(usize, Vec<bool>),
        adj_nodess: &[Vec<(usize, i128)>],
        memoize: &mut HashMap<(usize, Vec<bool>), Option<Vec<usize>>>,
        count: usize,
    ) -> Option<Vec<usize>> {
        if let Some(result) = memoize.get(param) {
            return result.clone();
        }
        let end = param.0;
        let mut visited = param.1.clone();
        visited[end] = true;
        let count = count + 1;
        if count == visited.len() {
            return Some(vec![end]);
        }
        let mut next_param = (0, visited);
        let mut result = None;
        for &(next_end, _) in &adj_nodess[end] {
            if !param.1[next_end] {
                next_param.0 = next_end;
                if let Some(mut path) = path(&next_param, adj_nodess, memoize, count) {
                    path.push(end);
                    result = Some(path);
                    break;
                }
            }
        }
        memoize.insert(next_param, result.clone());
        result
    }
    let mut param = (0, visited);
    for end in 0..adj_nodess.len() {
        param.0 = end;
        if let Some(path) = path(&param, adj_nodess, &mut memoize, 0) {
            return Some(path);
        }
    }
    None
}

mod max_flow {
    use std::collections::{HashMap, HashSet};

    pub type FlowIndex = (usize, usize, HashMap<(usize, usize), i128>, Vec<Vec<usize>>);

    fn get_flow(rem_flow: &HashMap<(usize, usize), i128>, in_node: usize, out_node: usize) -> i128 {
        *rem_flow.get(&(in_node, out_node)).unwrap_or(&0)
    }
    fn change_flow(
        rem_flow: &mut HashMap<(usize, usize), i128>,
        in_node: usize,
        out_node: usize,
        change: i128,
    ) {
        let new_cap = get_flow(rem_flow, in_node, out_node) + change;
        rem_flow.insert((in_node, out_node), new_cap);
    }

    /// O(m^2 log(max(weight)) + n)
    /// if all weights are the same: O(m + n)
    pub fn get_max_flow(
        adj_nodess: &[Vec<(usize, i128)>],
        start: usize,
        end: usize,
    ) -> (i128, FlowIndex) {
        // index flow amount
        let mut rem_flow = HashMap::with_capacity(adj_nodess.len());
        let mut bi_dir_adj_nodess = vec![HashSet::new(); adj_nodess.len()];
        for (node, adj_nodes) in adj_nodess.iter().enumerate() {
            for &(adj_node, weight) in adj_nodes {
                change_flow(&mut rem_flow, node, adj_node, weight);
                bi_dir_adj_nodess[node].insert(adj_node);
                bi_dir_adj_nodess[adj_node].insert(node);
            }
        }
        let adj_nodess = bi_dir_adj_nodess
            .into_iter()
            .map(|adj_nodes| adj_nodes.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // calculate flow
        let mut c = adj_nodess[start]
            .iter()
            .map(|&adj_node| get_flow(&rem_flow, start, adj_node))
            .max()
            .unwrap_or(0);
        fn find_path(
            c: i128,
            start: usize,
            end: usize,
            visited: &mut [bool],
            adj_nodess: &[Vec<usize>],
            rem_flow: &HashMap<(usize, usize), i128>,
        ) -> Option<Vec<usize>> {
            visited[start] = true;
            if start == end {
                return Some(vec![start]);
            }
            for &adj_node in &adj_nodess[start] {
                if !visited[adj_node] && get_flow(rem_flow, start, adj_node) >= c {
                    if let Some(mut path) =
                        find_path(c, adj_node, end, visited, adj_nodess, rem_flow)
                    {
                        path.push(start);
                        return Some(path);
                    }
                }
            }
            None
        }
        let mut max_flow = 0;
        while c > 0 {
            let mut visited = vec![false; adj_nodess.len()];
            if let Some(path) = find_path(c, start, end, &mut visited, &adj_nodess, &rem_flow) {
                let edges = path.iter().skip(1).zip(path.iter()).collect::<Vec<_>>();
                let flow_consumed = path
                    .iter()
                    .skip(1)
                    .zip(path.iter())
                    .map(|(&in_node, &out_node)| get_flow(&rem_flow, in_node, out_node))
                    .min()
                    .unwrap();
                for (&in_node, &out_node) in edges {
                    change_flow(&mut rem_flow, in_node, out_node, -flow_consumed);
                    change_flow(&mut rem_flow, out_node, in_node, flow_consumed);
                }
                max_flow += flow_consumed;
            } else {
                c /= 2;
            }
        }
        (max_flow, (start, end, rem_flow, adj_nodess))
    }

    /// O(m + n)
    pub fn get_min_cut((start, _end, rem_flow, adj_nodess): &FlowIndex) -> Vec<(usize, usize)> {
        let mut visited = vec![false; adj_nodess.len()];
        fn dfs(
            node: usize,
            visited: &mut [bool],
            adj_nodess: &[Vec<usize>],
            rem_flow: &HashMap<(usize, usize), i128>,
        ) {
            visited[node] = true;
            for &adj_node in &adj_nodess[node] {
                if !visited[adj_node] && get_flow(rem_flow, node, adj_node) > 0 {
                    dfs(adj_node, visited, adj_nodess, rem_flow);
                }
            }
        }
        dfs(*start, &mut visited, adj_nodess, rem_flow);
        let mut min_cut = Vec::new();
        for (node, &node_visited) in visited.iter().enumerate() {
            if node_visited {
                for &adj_node in &adj_nodess[node] {
                    if get_flow(rem_flow, node, adj_node) == 0 && !visited[adj_node] {
                        min_cut.push((node, adj_node));
                    }
                }
            }
        }
        min_cut
    }

    /// O(m + n)
    pub fn get_edge_disjoint_paths_count(
        adj_nodess: &[Vec<(usize, i128)>],
        start: usize,
        end: usize,
    ) -> (usize, FlowIndex) {
        let adj_nodess = adj_nodess
            .iter()
            .map(|adj_nodes| {
                adj_nodes
                    .iter()
                    .map(|&(adj_node, _)| (adj_node, 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let max_flow = get_max_flow(&adj_nodess, start, end);
        (max_flow.0 as usize, max_flow.1)
    }
    /// O(m + n)
    pub fn get_edge_disjoint_paths(
        adj_nodess: &[Vec<(usize, i128)>],
        (start, end, rem_flow, _): &FlowIndex,
    ) -> Vec<Vec<usize>> {
        let (start, end) = (*start, *end);
        // filter edge with non zero rem flow
        let mut adj_nodess = adj_nodess
            .iter()
            .enumerate()
            .map(|(node, adj_nodes)| {
                adj_nodes
                    .iter()
                    .filter_map(|&(adj_node, _)| {
                        if get_flow(rem_flow, node, adj_node) == 0 {
                            Some(adj_node)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // build paths
        let mut paths = Vec::new();
        while let Some(snd) = adj_nodess[start].pop() {
            let mut path = Vec::new();
            path.push(start);
            let mut node = snd;
            while node != end {
                path.push(node);
                node = adj_nodess[node].pop().unwrap();
            }
            path.push(end);
            paths.push(path);
        }
        paths
    }

    pub type DisjointNodeIndex = (FlowIndex, Vec<Vec<(usize, i128)>>);
    /// O(m + n)
    pub fn get_node_disjoint_paths_count(
        adj_nodess: &[Vec<(usize, i128)>],
        start: usize,
        end: usize,
    ) -> (usize, DisjointNodeIndex) {
        // i * 2 is in node while i * 2 + 1
        let mut adj_nodess_splitted = vec![Vec::new(); adj_nodess.len() * 2];
        for i in 0..adj_nodess.len() {
            adj_nodess_splitted[i * 2] = vec![(i * 2 + 1, 1)];
            adj_nodess_splitted[i * 2 + 1] = adj_nodess[i]
                .iter()
                .map(|&(adj_node, _)| (adj_node * 2, 1))
                .collect();
        }
        let max_flow = get_max_flow(&adj_nodess_splitted, start * 2, end * 2 + 1);
        (max_flow.0 as usize, (max_flow.1, adj_nodess_splitted))
    }
    /// O(m + n)
    pub fn get_node_disjoint_paths(
        (index, adj_nodess_splitted): &DisjointNodeIndex,
    ) -> Vec<Vec<usize>> {
        get_edge_disjoint_paths(adj_nodess_splitted, index)
            .into_iter()
            .map(|path| {
                let mut path = path.into_iter().map(|node| node / 2).collect::<Vec<_>>();
                path.dedup();
                path
            })
            .collect::<Vec<_>>()
    }
}
use max_flow::*;

pub struct DirectedGraph {
    adj_nodess: Vec<Vec<(usize, i128)>>,
    rev_adj_nodess: Vec<Vec<(usize, i128)>>,
    neg_edge_count: usize,
}
impl DirectedGraph {
    pub fn new() -> Self {
        Self {
            adj_nodess: Vec::new(),
            rev_adj_nodess: Vec::new(),
            neg_edge_count: 0,
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            adj_nodess: Vec::with_capacity(capacity),
            rev_adj_nodess: Vec::with_capacity(capacity),
            neg_edge_count: 0,
        }
    }
    /// O(m + n)
    pub fn from_edges(edges: Vec<(usize, usize, i128)>, node_count: usize) -> Self {
        let mut g = Self {
            adj_nodess: vec![Vec::new(); node_count],
            rev_adj_nodess: vec![Vec::new(); node_count],
            neg_edge_count: 0,
        };
        for edge in edges {
            g.add_edge(edge);
        }
        g
    }

    pub fn get_adj_nodess(&self) -> &Vec<Vec<(usize, i128)>> {
        &self.adj_nodess
    }
    pub fn get_rev_adj_nodess(&self) -> &Vec<Vec<(usize, i128)>> {
        &self.rev_adj_nodess
    }

    pub fn node_count(&self) -> usize {
        self.adj_nodess.len()
    }

    /// O(1)
    pub fn add_node(&mut self) -> usize {
        self.adj_nodess.push(Vec::new());
        self.rev_adj_nodess.push(Vec::new());
        self.adj_nodess.len() - 1
    }
    /// O(1)
    pub fn add_edge(&mut self, edge: (usize, usize, i128)) {
        self.adj_nodess[edge.0].push((edge.1, edge.2));
        self.rev_adj_nodess[edge.1].push((edge.0, edge.2));
        if edge.2 < 0 {
            self.neg_edge_count += 1;
        }
    }

    /// neg weight: O(mn)
    /// pos weight average: O(m + nlog(m))
    /// pos weight worst: O((m + n)log(m))
    pub fn get_shortest_path_lens(&self, start: usize) -> Option<Vec<i128>> {
        self.get_shortest_path_lens_till_stop(start, |_| false)
    }
    /// neg weight: O(mn)
    /// pos weight average: O(m + nlog(m))
    /// pos weight worst: O((m + n)log(m))
    pub fn get_shortest_path_lens_till_stop<F: Fn(usize) -> bool>(
        &self,
        start: usize,
        stop_when: F,
    ) -> Option<Vec<i128>> {
        if self.neg_edge_count > 0 {
            spfa(&self.adj_nodess, start)
        } else {
            Some(dijkstra(&self.adj_nodess, start, stop_when))
        }
    }

    /// O(n^3)
    pub fn get_all_shortest_path_lens(&self) -> Option<Vec<Vec<i128>>> {
        floyd_warshall(&self.adj_nodess)
    }

    /// O(n + m)
    pub fn reconstruct_shortest_path(
        &self,
        shortest_path_lens: &[i128],
        start: usize,
        end: usize,
    ) -> Option<Vec<usize>> {
        reconstruct_shortest_path(&self.rev_adj_nodess, shortest_path_lens, start, end)
    }
    /// O(n + m)
    pub fn reconstruct_all_shortest_path(
        &self,
        shortest_path_lens: &[i128],
        start: usize,
    ) -> DirectedGraph {
        reconstruct_all_shortest_path(&self.adj_nodess, shortest_path_lens, start)
    }

    /// O(n + m)
    pub fn get_topological_sort(&self, from: Option<usize>) -> Result<Vec<usize>, Vec<usize>> {
        let mut rev_sort = Vec::with_capacity(self.node_count());
        let mut states = vec![0; self.node_count()];
        fn have_cycle(
            current: usize,
            rev_sort: &mut Vec<usize>,
            states: &mut [u8],
            adj_nodess: &[Vec<(usize, i128)>],
        ) -> Option<Vec<usize>> {
            states[current] = 1;
            for &(adj_node, _) in &adj_nodess[current] {
                if states[adj_node] == 2 {
                    continue;
                }
                if states[adj_node] == 1 {
                    return Some(vec![adj_node, current]);
                }
                if let Some(mut cycle) = have_cycle(adj_node, rev_sort, states, adj_nodess) {
                    if cycle[0] != *cycle.last().unwrap() {
                        // cycle isn't complete yet
                        cycle.push(current);
                    }
                    return Some(cycle);
                }
            }
            states[current] = 2;
            rev_sort.push(current);
            None
        }
        let origins = if let Some(from) = from {
            from..(from + 1)
        } else {
            0..states.len()
        };
        for node in origins {
            if states[node] == 0 {
                let cycle = have_cycle(node, &mut rev_sort, &mut states, &self.adj_nodess);
                if let Some(cycle) = cycle {
                    return Err(cycle.into_iter().skip(1).rev().collect());
                }
            }
        }
        Ok(rev_sort.into_iter().rev().collect())
    }

    /// O(n + m)
    pub fn get_strongly_connected_components(&self) -> Vec<HashSet<usize>> {
        let rev_adj_nodess = self.get_rev_adj_nodess();
        let adj_nodess = &self.adj_nodess;

        // first phase
        fn dfs(
            node: usize,
            adj_nodess: &[Vec<(usize, i128)>],
            seen: &mut [bool],
            processed_order: &mut Vec<usize>,
        ) {
            seen[node] = true;
            for &(adj_node, _weight) in &adj_nodess[node] {
                if !seen[adj_node] {
                    dfs(adj_node, adj_nodess, seen, processed_order);
                }
            }
            processed_order.push(node);
        }
        let mut seen = vec![false; adj_nodess.len()];
        let mut processed_order = Vec::with_capacity(adj_nodess.len());
        for node in 0..adj_nodess.len() {
            if !seen[node] {
                dfs(node, adj_nodess, &mut seen, &mut processed_order);
            }
        }

        // second phase
        fn dfs2(
            node: usize,
            rev_adj_nodess: &[Vec<(usize, i128)>],
            added: &mut [bool],
            members: &mut HashSet<usize>,
        ) {
            members.insert(node);
            added[node] = true;
            for &(adj_node, _weight) in &rev_adj_nodess[node] {
                if !added[adj_node] {
                    dfs2(adj_node, rev_adj_nodess, added, members);
                }
            }
        }
        let mut added = vec![false; adj_nodess.len()];
        let mut strongly_connected_components = Vec::new();
        for &node in processed_order.iter().rev() {
            if !added[node] {
                let mut members = HashSet::new();
                dfs2(node, rev_adj_nodess, &mut added, &mut members);
                strongly_connected_components.push(members);
            }
        }
        strongly_connected_components
    }

    /// O(n)
    pub fn get_eulerian_start_end(&self) -> Option<(usize, usize)> {
        if self.adj_nodess.is_empty()
            || DepthFirstIter::new(&self.adj_nodess, 0).count() != self.node_count()
        {
            return None;
        }
        let extra_out = self
            .adj_nodess
            .iter()
            .enumerate()
            .zip(self.rev_adj_nodess.iter())
            .filter_map(|((node, adj_nodes), rev_adj_nodes)| {
                let extra_out = adj_nodes.len() as i128 - rev_adj_nodes.len() as i128;
                if extra_out != 0 {
                    Some((node, extra_out))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if extra_out.is_empty() {
            Some((0, 0))
        } else if extra_out.len() == 2 && extra_out[0].1 == 1 && extra_out[1].1 == -1 {
            Some((extra_out[0].0, extra_out[1].0))
        } else if extra_out.len() == 2 && extra_out[0].1 == -1 && extra_out[1].1 == 1 {
            Some((extra_out[1].0, extra_out[0].0))
        } else {
            None
        }
    }
    /// O(m + n)
    pub fn get_eulerian_path(&self) -> Option<Vec<usize>> {
        let (start, end) = if let Some((start, end)) = self.get_eulerian_start_end() {
            (start, end)
        } else {
            return None;
        };
        let mut next_edge_to_walk = vec![0; self.node_count()];
        let mut path = Vec::with_capacity(self.node_count());
        fn build_path(
            start: usize,
            end: usize,
            adj_nodess: &[Vec<(usize, i128)>],
            path: &mut Vec<usize>,
            next_edge_to_walk: &mut [usize],
        ) {
            // build sub_path / cycle
            let mut sub_path = Vec::new();
            let mut node = start;
            if let Some(&(next_node, _)) = adj_nodess[node].get(next_edge_to_walk[node]) {
                sub_path.push(node);
                next_edge_to_walk[node] += 1;
                node = next_node;
            } else {
                path.push(node);
                return;
            }
            while node != end {
                let (next_node, _) = adj_nodess[node][next_edge_to_walk[node]];
                sub_path.push(node);
                next_edge_to_walk[node] += 1;
                node = next_node;
            }
            sub_path.push(node);

            // build path
            for node in sub_path {
                build_path(node, node, adj_nodess, path, next_edge_to_walk);
            }
        }
        build_path(
            start,
            end,
            &self.adj_nodess,
            &mut path,
            &mut next_edge_to_walk,
        );
        Some(path)
    }

    /// O(2^n * n^2)
    pub fn get_hamiltonian_path(&self) -> Option<Vec<usize>> {
        hamiltonian_path(&self.adj_nodess)
    }

    /// O(m^2 log(max(weight)) + n)
    /// if all weights are the same: O(m + n)
    pub fn get_max_flow(&self, start: usize, end: usize) -> (i128, FlowIndex) {
        get_max_flow(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_min_cut(&self, index: &FlowIndex) -> Vec<(usize, usize)> {
        get_min_cut(index)
    }

    /// O(m + n)
    pub fn get_edge_disjoint_paths_count(&self, start: usize, end: usize) -> (usize, FlowIndex) {
        get_edge_disjoint_paths_count(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_edge_disjoint_paths(&self, index: &FlowIndex) -> Vec<Vec<usize>> {
        get_edge_disjoint_paths(&self.adj_nodess, index)
    }

    /// O(m + n)
    pub fn get_node_disjoint_paths_count(
        &self,
        start: usize,
        end: usize,
    ) -> (usize, DisjointNodeIndex) {
        get_node_disjoint_paths_count(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_node_disjoint_paths(&self, index: &DisjointNodeIndex) -> Vec<Vec<usize>> {
        get_node_disjoint_paths(index)
    }
}

#[cfg(test)]
mod test_directed_graph {
    use std::collections::HashSet;
    use std::iter::FromIterator;

    use super::*;

    // cphb p.g. 124
    const EDGES: &[(usize, usize, i128)] = &[
        (0, 1, 5),
        (0, 2, 3),
        (0, 3, 7),
        (1, 0, 5),
        (1, 3, 3),
        (1, 4, 2),
        (2, 0, 3),
        (2, 3, 1),
        (3, 0, 7),
        (3, 1, 3),
        (3, 2, 1),
        (3, 4, 2),
        (4, 1, 2),
        (4, 3, 2),
    ];

    #[test]
    fn dijkstra() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(g.get_shortest_path_lens(0), Some(vec![0, 5, 3, 4, 6]),);
    }

    #[test]
    fn spfa() {
        let mut edges = EDGES.to_vec();
        edges.push((2, 4, -1));
        let g = DirectedGraph::from_edges(edges, 5);
        assert_eq!(g.get_shortest_path_lens(0), Some(vec![0, 4, 3, 4, 2]),);
    }

    #[test]
    fn spfa_negative_cycle() {
        let mut edges = EDGES.to_vec();
        edges.push((2, 4, -4));
        let g = DirectedGraph::from_edges(edges, 5);
        assert_eq!(g.get_shortest_path_lens(1), None);
    }

    #[test]
    fn spfa_negative_cycle_no_false_positive() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 0),
                (1, 2, 0),
                (2, 3, -1),
                (3, 4, 0),
                (4, 5, 0),
                (5, 0, 1),
            ],
            6,
        );
        assert_eq!(g.get_shortest_path_lens(0), Some(vec![0, 0, 0, -1, -1, -1]));
    }

    #[test]
    fn floyd_warshall() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(
            g.get_all_shortest_path_lens(),
            Some(vec![
                vec![0, 5, 3, 4, 6],
                vec![5, 0, 4, 3, 2],
                vec![3, 4, 0, 1, 3],
                vec![4, 3, 1, 0, 2],
                vec![6, 2, 3, 2, 0],
            ]),
        );
    }

    #[test]
    fn floyd_warshall_negative_cycle() {
        let mut edges = EDGES.to_vec();
        edges.push((2, 4, -4));
        edges.push((4, 2, -4));
        let g = DirectedGraph::from_edges(edges, 5);
        assert_eq!(g.get_all_shortest_path_lens(), None);
    }

    #[test]
    fn floyd_warshall_negative_cycle_no_false_positive() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 0),
                (1, 2, 0),
                (2, 3, -1),
                (3, 4, 0),
                (4, 5, 0),
                (5, 0, 1),
            ],
            6,
        );
        assert_eq!(
            g.get_all_shortest_path_lens(),
            Some(vec![
                vec![0, 0, 0, -1, -1, -1],
                vec![0, 0, 0, -1, -1, -1],
                vec![0, 0, 0, -1, -1, -1],
                vec![1, 1, 1, 0, 0, 0],
                vec![1, 1, 1, 0, 0, 0],
                vec![1, 1, 1, 0, 0, 0],
            ]),
        );
    }

    #[test]
    fn reconstruct_shortest_path() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        let shortest_path_lens = g.get_shortest_path_lens(0).unwrap();
        assert_eq!(
            g.reconstruct_shortest_path(&shortest_path_lens, 0, 4),
            Some(vec![0, 2, 3, 4]),
        );
    }

    #[test]
    fn reconstruct_shortest_path_negative_edge() {
        let mut edges = EDGES.to_vec();
        edges.push((2, 4, -1));
        let g = DirectedGraph::from_edges(edges, 5);
        let shortest_path_lens = g.get_shortest_path_lens(0).unwrap();
        assert_eq!(
            g.reconstruct_shortest_path(&shortest_path_lens, 0, 4),
            Some(vec![0, 2, 4]),
        );
    }

    #[test]
    fn reconstruct_all_shortest_path() {
        let mut edges = EDGES.to_vec();
        edges.push((2, 4, -1));
        let g = DirectedGraph::from_edges(edges, 5);
        let shortest_path_lens = g.get_shortest_path_lens(0).unwrap();
        assert_eq!(
            g.reconstruct_all_shortest_path(&shortest_path_lens, 0)
                .get_adj_nodess()
                .iter()
                .map(|adj_nodes| { HashSet::from_iter(adj_nodes) })
                .collect::<Vec<HashSet<&(usize, i128)>>>(),
            vec![
                HashSet::from_iter(&[(2, 3)]),
                HashSet::from_iter(&[]),
                HashSet::from_iter(&[(3, 1), (4, -1)]),
                HashSet::from_iter(&[]),
                HashSet::from_iter(&[(1, 2), (3, 2)]),
            ],
        );
    }

    #[test]
    fn topological_sort() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (2, 5, 1),
                (3, 0, 1),
                (3, 4, 1),
                (4, 1, 1),
                (4, 2, 1),
            ],
            6,
        );
        assert_eq!(g.get_topological_sort(None), Ok(vec![3, 4, 0, 1, 2, 5]));
    }

    #[test]
    fn topological_sort_with_start() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (2, 5, 1),
                (3, 0, 1),
                (3, 4, 1),
                (4, 3, 1),
                (4, 1, 1),
                (4, 2, 1),
            ],
            6,
        );
        assert_eq!(g.get_topological_sort(Some(0)), Ok(vec![0, 1, 2, 5]));
    }

    #[test]
    fn cycle_detection() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (2, 5, 1),
                (3, 0, 1),
                (3, 4, 1),
                (4, 1, 1),
                (4, 2, 1),
                (5, 4, 1),
            ],
            6,
        );
        assert_eq!(g.get_topological_sort(None), Err(vec![1, 2, 5, 4]));
    }

    #[test]
    fn get_strongly_connected_components() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (0, 3, 1),
                (1, 0, 1),
                (1, 4, 1),
                (2, 1, 1),
                (2, 6, 1),
                (4, 3, 1),
                (5, 4, 1),
                (5, 2, 1),
                (6, 5, 1),
            ],
            7,
        );
        let mut components = g.get_strongly_connected_components();
        components.sort_by_key(|c| c.iter().min().unwrap().to_owned());
        assert_eq!(
            components,
            vec![
                [0, 1].iter().cloned().collect::<HashSet<_>>(),
                [2, 5, 6].iter().cloned().collect::<HashSet<_>>(),
                [3].iter().cloned().collect::<HashSet<_>>(),
                [4].iter().cloned().collect::<HashSet<_>>(),
            ],
        );
    }

    #[test]
    fn get_eulerian_start_end() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (2, 4, 1),
                (3, 0, 1),
                (4, 3, 1),
            ],
            5,
        );
        assert_eq!(g.get_eulerian_start_end(), Some((1, 4)));
    }

    #[test]
    fn no_eulerian_path() {
        let g = DirectedGraph::from_edges(
            vec![(0, 1, 1), (1, 2, 1), (1, 4, 1), (2, 4, 1), (3, 0, 1)],
            5,
        );
        assert_eq!(g.get_eulerian_start_end(), None);
    }

    #[test]
    fn get_eulerian_path() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (2, 4, 1),
                (3, 0, 1),
                (4, 3, 1),
            ],
            5,
        );
        assert_eq!(g.get_eulerian_path(), Some(vec![1, 4, 3, 0, 1, 2, 4]))
    }

    #[test]
    fn get_eulerian_cycle() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (2, 4, 1),
                (3, 0, 1),
                (4, 1, 1),
                (4, 3, 1),
            ],
            5,
        );
        assert_eq!(g.get_eulerian_path(), Some(vec![0, 1, 2, 4, 1, 4, 3, 0]))
    }

    #[test]
    fn get_hamiltonian_path() {
        let g = DirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(g.get_hamiltonian_path(), Some(vec![2, 3, 4, 1, 0]));
    }

    #[test]
    fn max_flow() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 5),
                (0, 3, 4),
                (1, 2, 6),
                (2, 4, 8),
                (2, 5, 5),
                (3, 1, 3),
                (3, 4, 1),
                (4, 5, 2),
            ],
            6,
        );
        assert_eq!(g.get_max_flow(0, 5).0, 7);
    }

    #[test]
    fn min_cut() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 5),
                (0, 3, 4),
                (1, 2, 6),
                (2, 4, 8),
                (2, 5, 5),
                (3, 1, 3),
                (3, 4, 1),
                (4, 5, 2),
            ],
            6,
        );
        let index = g.get_max_flow(0, 5).1;
        assert_eq!(g.get_min_cut(&index), vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn get_edge_disjoint_paths() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 5),
                (0, 3, 4),
                (1, 3, 6),
                (2, 1, 9),
                (2, 4, 8),
                (2, 5, 5),
                (3, 2, 3),
                (3, 4, 1),
                (4, 5, 2),
            ],
            6,
        );
        let (count, index) = g.get_edge_disjoint_paths_count(0, 5);
        assert_eq!(count, 2);
        assert_eq!(
            g.get_edge_disjoint_paths(&index),
            vec![vec![0, 3, 4, 5], vec![0, 1, 3, 2, 5]],
        );
    }

    #[test]
    fn get_node_disjoint_paths() {
        let g = DirectedGraph::from_edges(
            vec![
                (0, 1, 5),
                (0, 3, 4),
                (1, 3, 6),
                (2, 1, 9),
                (2, 4, 8),
                (2, 5, 5),
                (3, 2, 3),
                (3, 4, 1),
                (4, 5, 2),
            ],
            6,
        );
        let (count, index) = g.get_node_disjoint_paths_count(0, 5);
        assert_eq!(count, 1);
        let paths = g.get_node_disjoint_paths(&index);
        let adj_nodess = g.get_adj_nodess();
        for path in paths {
            let mut in_path = vec![false; g.node_count()];
            in_path[path[0]] = true;
            for (&node1, &node2) in path.iter().zip(path.iter().skip(1)) {
                assert!(!in_path[node2]);
                in_path[node2] = true;
                assert!(adj_nodess[node1].iter().any(|&(n, _)| n == node2));
            }
        }
    }
}

pub struct UndirectedGraph {
    adj_nodess: Vec<Vec<(usize, i128)>>,
    neg_edge_count: usize,
}
impl UndirectedGraph {
    pub fn new() -> Self {
        Self {
            adj_nodess: Vec::new(),
            neg_edge_count: 0,
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            adj_nodess: Vec::with_capacity(capacity),
            neg_edge_count: 0,
        }
    }
    /// O(m + n)
    pub fn from_edges(edges: Vec<(usize, usize, i128)>, node_count: usize) -> Self {
        let mut g = Self {
            adj_nodess: vec![Vec::new(); node_count],
            neg_edge_count: 0,
        };
        for edge in edges {
            g.add_edge(edge);
        }
        g
    }

    pub fn get_adj_nodess(&self) -> &Vec<Vec<(usize, i128)>> {
        &self.adj_nodess
    }
    pub fn node_count(&self) -> usize {
        self.adj_nodess.len()
    }

    /// O(1)
    pub fn add_node(&mut self) -> usize {
        self.adj_nodess.push(Vec::new());
        self.adj_nodess.len() - 1
    }
    /// O(1)
    pub fn add_edge(&mut self, edge: (usize, usize, i128)) {
        self.adj_nodess[edge.0].push((edge.1, edge.2));
        self.adj_nodess[edge.1].push((edge.0, edge.2));
        if edge.2 < 0 {
            self.neg_edge_count += 1;
        }
    }

    /// average: O(m + nlog(m))
    /// worst: O((m + n)log(m))
    pub fn get_shortest_path_lens(&self, start: usize) -> Option<Vec<i128>> {
        self.get_shortest_path_lens_till_stop(start, |_| false)
    }
    /// average: O(m + nlog(m))
    /// worst: O((m + n)log(m))
    pub fn get_shortest_path_lens_till_stop<F: Fn(usize) -> bool>(
        &self,
        start: usize,
        stop_when: F,
    ) -> Option<Vec<i128>> {
        if self.neg_edge_count == 0 {
            Some(dijkstra(&self.adj_nodess, start, stop_when))
        } else {
            None
        }
    }

    /// O(n^3)
    pub fn get_all_shortest_path_lens(&self) -> Option<Vec<Vec<i128>>> {
        if self.neg_edge_count > 0 {
            None
        } else {
            floyd_warshall(&self.adj_nodess)
        }
    }

    /// O(n + m)
    pub fn reconstruct_shortest_path(
        &self,
        shortest_path_lens: &[i128],
        start: usize,
        end: usize,
    ) -> Option<Vec<usize>> {
        reconstruct_shortest_path(&self.adj_nodess, shortest_path_lens, start, end)
    }
    /// O(n + m)
    pub fn reconstruct_all_shortest_path(
        &self,
        shortest_path_lens: &[i128],
        start: usize,
    ) -> DirectedGraph {
        reconstruct_all_shortest_path(&self.adj_nodess, shortest_path_lens, start)
    }

    /// average: O(m + nlog(m))
    /// worst: O((m + n)log(m))
    pub fn get_min_spanning_tree(&self) -> Tree {
        let node_count = self.node_count();
        if node_count == 0 {
            return Tree::new();
        }
        let mut added = vec![false; node_count];
        added[0] = true;
        let mut edges = Vec::with_capacity(node_count);
        let mut queue =
            BinaryHeap::from_iter(self.adj_nodess[0].iter().filter_map(|&(adj_node, weight)| {
                if 0 == adj_node {
                    None
                } else {
                    Some(Reverse((weight, 0, adj_node)))
                }
            }));
        while let Some(Reverse((weight, added_node, new_node))) = queue.pop() {
            if !added[new_node] {
                added[new_node] = true;
                edges.push((added_node, new_node, weight));
                let new_edges =
                    self.adj_nodess[new_node]
                        .iter()
                        .filter_map(|&(adj_node, weight)| {
                            if added[adj_node] {
                                None
                            } else {
                                Some(Reverse((weight, new_node, adj_node)))
                            }
                        });

                for new_edge in new_edges {
                    queue.push(new_edge);
                }
            }
        }
        Tree::from_edges(edges, node_count)
    }

    /// O(n)
    pub fn get_eulerian_start_end(&self) -> Option<(usize, usize)> {
        if self.adj_nodess.is_empty()
            || DepthFirstIter::new(&self.adj_nodess, 0).count() != self.node_count()
        {
            return None;
        }
        let odd_degree_nodes = self
            .adj_nodess
            .iter()
            .enumerate()
            .filter_map(|(index, adj_nodes)| {
                if adj_nodes.len() % 2 == 1 {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if odd_degree_nodes.len() == 2 {
            Some((odd_degree_nodes[0], odd_degree_nodes[1]))
        } else if odd_degree_nodes.is_empty() {
            Some((0, 0))
        } else {
            None
        }
    }
    /// O(m + n)
    pub fn get_eulerian_path(&self) -> Option<Vec<usize>> {
        let (start, end) = if let Some((start, end)) = self.get_eulerian_start_end() {
            (start, end)
        } else {
            return None;
        };
        // index edges to avoid duplication
        let mut remaining_edges = MultiSet::new();
        for (node, adj_nodes) in self.adj_nodess.iter().enumerate() {
            let mut self_edge_count = 0;
            for &(adj_node, _) in adj_nodes {
                match node.cmp(&adj_node) {
                    Ordering::Less => remaining_edges.insert((node, adj_node)),
                    Ordering::Equal => self_edge_count += 1,
                    _ => {}
                }
            }
            for _ in 0..self_edge_count / 2 {
                remaining_edges.insert((node, node));
            }
        }

        let mut next_edge_to_walk = vec![0; self.node_count()];
        let mut path = Vec::with_capacity(self.node_count());
        fn ot(a: usize, b: usize) -> (usize, usize) {
            (a.min(b), a.max(b))
        }
        fn build_path(
            start: usize,
            end: usize,
            adj_nodess: &[Vec<(usize, i128)>],
            path: &mut Vec<usize>,
            next_edge_to_walk: &mut [usize],
            remaining_edges: &mut MultiSet<(usize, usize)>,
        ) {
            // build sub_path / cycle
            let mut sub_path = Vec::new();
            let mut node = start;

            // check node has unused edges
            while {
                if next_edge_to_walk[node] < adj_nodess[node].len() {
                    let edge_index = ot(adj_nodess[node][next_edge_to_walk[node]].0, node);
                    remaining_edges.count(&edge_index) == 0
                } else {
                    false
                }
            } {
                next_edge_to_walk[node] += 1;
            }

            if let Some(&(next_node, _)) = adj_nodess[node].get(next_edge_to_walk[node]) {
                sub_path.push(node);
                next_edge_to_walk[node] += 1;
                remaining_edges.remove(&ot(node, next_node));
                node = next_node;
            } else {
                path.push(node);
                return;
            }

            // build sub_path
            while node != end {
                // check node has unused edges
                while {
                    if next_edge_to_walk[node] < adj_nodess[node].len() {
                        let edge_index = ot(adj_nodess[node][next_edge_to_walk[node]].0, node);
                        remaining_edges.count(&edge_index) == 0
                    } else {
                        false
                    }
                } {
                    next_edge_to_walk[node] += 1;
                }

                let (next_node, _) = adj_nodess[node][next_edge_to_walk[node]];
                sub_path.push(node);
                next_edge_to_walk[node] += 1;
                remaining_edges.remove(&ot(node, next_node));
                node = next_node;
            }
            sub_path.push(node);

            // build path
            for node in sub_path {
                build_path(
                    node,
                    node,
                    adj_nodess,
                    path,
                    next_edge_to_walk,
                    remaining_edges,
                );
            }
        }
        build_path(
            start,
            end,
            &self.adj_nodess,
            &mut path,
            &mut next_edge_to_walk,
            &mut remaining_edges,
        );
        Some(path)
    }

    /// O(2^n * n^2)
    pub fn get_hamiltonian_path(&self) -> Option<Vec<usize>> {
        hamiltonian_path(&self.adj_nodess)
    }

    /// O(m^2 log(max(weight)) + n)
    /// if all weights are the same: O(m + n)
    pub fn get_max_flow(&self, start: usize, end: usize) -> (i128, FlowIndex) {
        get_max_flow(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_min_cut(&self, index: &FlowIndex) -> Vec<(usize, usize)> {
        get_min_cut(index)
    }

    /// O(m + n)
    pub fn get_edge_disjoint_paths_count(&self, start: usize, end: usize) -> (usize, FlowIndex) {
        get_edge_disjoint_paths_count(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_edge_disjoint_paths(&self, index: &FlowIndex) -> Vec<Vec<usize>> {
        get_edge_disjoint_paths(&self.adj_nodess, index)
    }

    /// O(m + n)
    pub fn get_node_disjoint_paths_count(
        &self,
        start: usize,
        end: usize,
    ) -> (usize, DisjointNodeIndex) {
        get_node_disjoint_paths_count(&self.adj_nodess, start, end)
    }

    /// O(m + n)
    pub fn get_node_disjoint_paths(&self, index: &DisjointNodeIndex) -> Vec<Vec<usize>> {
        get_node_disjoint_paths(index)
    }
}

#[cfg(test)]
mod test_undirected_graph {
    use std::collections::HashSet;
    use std::iter::FromIterator;

    use super::UndirectedGraph;

    // cphb p.g. 124
    const EDGES: &[(usize, usize, i128)] = &[
        (0, 1, 5),
        (0, 2, 3),
        (0, 3, 7),
        (1, 3, 3),
        (1, 4, 2),
        (2, 3, 1),
        (3, 4, 2),
    ];

    #[test]
    fn dijkstra() {
        let g = UndirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(g.get_shortest_path_lens(0), Some(vec![0, 5, 3, 4, 6]),);
    }

    #[test]
    fn shortest_path_with_neg_edge() {
        let mut g = UndirectedGraph::from_edges(EDGES.to_vec(), 5);
        g.add_edge((1, 2, -1));
        assert_eq!(g.get_shortest_path_lens(0), None);
    }

    #[test]
    fn reconstruct_shortest_path() {
        let g = UndirectedGraph::from_edges(EDGES.to_vec(), 5);
        let shortest_path_lens = g.get_shortest_path_lens(0).unwrap();
        assert_eq!(
            g.reconstruct_shortest_path(&shortest_path_lens, 0, 4),
            Some(vec![0, 2, 3, 4])
        );
    }

    #[test]
    fn reconstruct_all_shortest_path() {
        let g = UndirectedGraph::from_edges(EDGES.to_vec(), 5);
        let shortest_path_lens = g.get_shortest_path_lens(0).unwrap();
        assert_eq!(
            g.reconstruct_all_shortest_path(&shortest_path_lens, 0)
                .get_adj_nodess()
                .iter()
                .map(|adj_nodes| { HashSet::from_iter(adj_nodes) })
                .collect::<Vec<HashSet<&(usize, i128)>>>(),
            vec![
                HashSet::from_iter(&[(1, 5), (2, 3)]),
                HashSet::from_iter(&[]),
                HashSet::from_iter(&[(3, 1)]),
                HashSet::from_iter(&[(4, 2)]),
                HashSet::from_iter(&[]),
            ],
        );
    }

    #[test]
    fn min_spanning_tree() {
        let g = UndirectedGraph::from_edges(EDGES.to_vec(), 5); // (0, 2, 3), (1, 4, 2), (2, 3, 1), (3, 4, 2)
        assert_eq!(
            g.get_min_spanning_tree()
                .get_adj_nodess()
                .iter()
                .map(|adj_nodes| { HashSet::from_iter(adj_nodes) })
                .collect::<Vec<HashSet<&(usize, i128)>>>(),
            vec![
                HashSet::from_iter(&[(2, 3)]),
                HashSet::from_iter(&[(4, 2)]),
                HashSet::from_iter(&[(0, 3), (3, 1)]),
                HashSet::from_iter(&[(2, 1), (4, 2)]),
                HashSet::from_iter(&[(1, 2), (3, 2)]),
            ],
        );
    }

    #[test]
    fn floyd_warshall() {
        let g = UndirectedGraph::from_edges(EDGES.to_vec(), 5);
        assert_eq!(
            g.get_all_shortest_path_lens(),
            Some(vec![
                vec![0, 5, 3, 4, 6],
                vec![5, 0, 4, 3, 2],
                vec![3, 4, 0, 1, 3],
                vec![4, 3, 1, 0, 2],
                vec![6, 2, 3, 2, 0],
            ]),
        );
    }

    #[test]
    fn get_eulerian_start_end() {
        let g = UndirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (2, 4, 1),
                (3, 0, 1),
                (4, 3, 1),
            ],
            5,
        );
        let eulerian_start_end = g.get_eulerian_start_end().unwrap();
        assert_eq!(
            [eulerian_start_end.0, eulerian_start_end.1].iter().collect::<HashSet<_>>(),
            [1, 4].iter().collect::<HashSet<_>>(),
        );
    }

    #[test]
    fn no_eulerian_path() {
        let g = UndirectedGraph::from_edges(
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (2, 4, 1),
                (3, 0, 1),
                (4, 3, 1),
                (0, 2, 1),
            ],
            5,
        );
        assert_eq!(g.get_eulerian_start_end(), None);
    }

    #[test]
    fn get_eulerian_path() {
        let mut edges = vec![
            (0, 1, 1),
            (1, 2, 1),
            (1, 4, 1),
            (2, 4, 1),
            (3, 0, 1),
            (4, 3, 1),
        ];
        let g = UndirectedGraph::from_edges(edges.clone(), 5);
        let path = g.get_eulerian_path().unwrap();
        for (node1, node2) in path.iter().zip(path.iter().skip(1)) {
            let mut to_remove = edges.len();
            for (i, (node_1, node_2, _)) in edges.iter().enumerate() {
                if (node1 == node_1 && node2 == node_2) || (node1 == node_2 && node2 == node_1) {
                    to_remove = i;
                    break;
                }
            }
            edges.remove(to_remove);
        }
        assert!(edges.is_empty());
    }

    #[test]
    fn get_eulerian_cycle() {
        let mut edges = vec![
            (0, 1, 1),
            (1, 2, 1),
            (1, 4, 1),
            (2, 4, 1),
            (3, 0, 1),
            (4, 1, 1),
            (4, 3, 1),
        ];
        let g = UndirectedGraph::from_edges(edges.clone(), 5);
        let path = g.get_eulerian_path().unwrap();
        for (node1, node2) in path.iter().zip(path.iter().skip(1)) {
            let mut to_remove = edges.len();
            for (i, (node_1, node_2, _)) in edges.iter().enumerate() {
                if (node1 == node_1 && node2 == node_2) || (node1 == node_2 && node2 == node_1) {
                    to_remove = i;
                    break;
                }
            }
            edges.remove(to_remove);
        }
        assert!(edges.is_empty());
    }

    #[test]
    fn no_hamiltonian_path() {
        let g = UndirectedGraph::from_edges(vec![(0, 1, 1), (0, 2, 1), (0, 3, 1)], 4);
        assert_eq!(g.get_hamiltonian_path(), None);
    }
}

pub type MaxMatchIndex = (FlowIndex, DirectedGraph);
/// O(n)
pub fn get_max_matchings_count(
    matches: &[(usize, usize)],
    node_count: usize,
) -> (usize, MaxMatchIndex) {
    let fst_nodes = matches.iter().map(|&(fst, _)| fst).collect::<HashSet<_>>();
    let snd_nodes = matches.iter().map(|&(_, snd)| snd).collect::<HashSet<_>>();
    assert!(
        fst_nodes.is_disjoint(&snd_nodes),
        "invalide format, first nodes and second nodes aren't disjoint",
    );

    let start = node_count;
    let end = start + 1;
    let mid_edges = matches.iter().map(|&(fst, snd)| (fst, snd, 1));
    let start_edges = fst_nodes.into_iter().map(|fst| (start, fst, 1));
    let end_edges = snd_nodes.into_iter().map(|snd| (snd, end, 1));
    let graph = DirectedGraph::from_edges(
        start_edges.chain(mid_edges).chain(end_edges).collect(),
        node_count + 2,
    );
    let (count, flow_index) = graph.get_edge_disjoint_paths_count(start, end);
    (count, (flow_index, graph))
}
/// O(n)
pub fn get_max_matchings((flow_index, graph): &MaxMatchIndex) -> Vec<(usize, usize)> {
    graph
        .get_edge_disjoint_paths(flow_index)
        .into_iter()
        .map(|path| (path[1], path[2]))
        .collect()
}

#[cfg(test)]
mod test_max_matching {
    use super::*;

    #[test]
    fn max_matchings() {
        let edges = &[(0, 4), (1, 6), (2, 4), (2, 5), (2, 7), (3, 6)];
        let (count, index) = get_max_matchings_count(edges, 8);
        assert_eq!(count, 3);
        let max_matchings = get_max_matchings(&index);
        assert_eq!(
            max_matchings
                .iter()
                .map(|&(fst, _)| fst)
                .collect::<HashSet<_>>()
                .len(),
            3,
        );
        assert_eq!(
            max_matchings
                .iter()
                .map(|&(_, snd)| snd)
                .collect::<HashSet<_>>()
                .len(),
            3,
        );
        assert!(edges.iter().collect::<HashSet<_>>().is_superset(&max_matchings.iter().collect::<HashSet<_>>()));
    }
}

pub struct Tree {
    adj_nodess: Vec<Vec<(usize, i128)>>,
}
impl Tree {
    pub fn new() -> Self {
        Self {
            adj_nodess: Vec::new(),
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            adj_nodess: Vec::with_capacity(capacity),
        }
    }
    /// O(n)
    pub fn from_edges(edges: Vec<(usize, usize, i128)>, node_count: usize) -> Self {
        assert!(
            edges.len() == node_count - 1,
            "Incorrect amonut of edges for a tree",
        );
        let mut added = vec![false; node_count];
        let mut adj_nodess = vec![Vec::new(); node_count];
        for (node1, node2, weight) in edges {
            assert!(
                !(added[node1] && added[node2]),
                "There's a cycle in your tree",
            );
            added[node1] = true;
            added[node2] = true;
            adj_nodess[node1].push((node2, weight));
            adj_nodess[node2].push((node1, weight));
        }
        Tree { adj_nodess }
    }

    pub fn get_adj_nodess(&self) -> &Vec<Vec<(usize, i128)>> {
        &self.adj_nodess
    }

    pub fn node_count(&self) -> usize {
        self.adj_nodess.len()
    }

    /// O(1)
    pub fn add_node(&mut self, (connected_node, weight): (usize, i128)) -> usize {
        let new_node = self.adj_nodess.len();
        self.adj_nodess[connected_node].push((new_node, weight));
        self.adj_nodess.push(vec![(connected_node, weight)]);
        new_node
    }

    /// O(n)
    pub fn get_diameter(&self) -> i128 {
        if self.adj_nodess.is_empty() {
            return 0;
        }
        let mut dist_to_zero = vec![std::i128::MAX; self.node_count()];
        dist_to_zero[0] = 0;
        for node in DepthFirstIter::new(self.get_adj_nodess(), 0).skip(1) {
            dist_to_zero[node] = self.adj_nodess[node]
                .iter()
                .find_map(|&(adj_node, weight)| {
                    if dist_to_zero[adj_node] < std::i128::MAX {
                        Some(dist_to_zero[adj_node] + weight)
                    } else {
                        None
                    }
                })
                .unwrap();
        }
        let start = dist_to_zero
            .into_iter()
            .enumerate()
            .max_by_key(|&(_, dist)| dist)
            .unwrap()
            .0;
        let mut dist_to_start = vec![std::i128::MAX; self.node_count()];
        dist_to_start[start] = 0;
        for node in DepthFirstIter::new(self.get_adj_nodess(), start).skip(1) {
            dist_to_start[node] = self.adj_nodess[node]
                .iter()
                .find_map(|&(adj_node, weight)| {
                    if dist_to_start[adj_node] < std::i128::MAX {
                        Some(dist_to_start[adj_node] + weight)
                    } else {
                        None
                    }
                })
                .unwrap();
        }
        dist_to_start
            .into_iter()
            .enumerate()
            .max_by_key(|&(_, dist)| dist)
            .unwrap()
            .1
    }

    /// O(n)
    pub fn get_longest_path_lens(&self) -> Vec<i128> {
        if self.adj_nodess.is_empty() {
            return Vec::new();
        }
        if self.node_count() == 1 {
            return vec![0];
        }
        fn get_longest_path_down(
            node: usize,
            parent: usize,
            adj_nodess: &[Vec<(usize, i128)>],
            longest_2_paths_down: &mut [(i128, usize, i128)],
        ) -> i128 {
            let mut longest_path = 0;
            let mut longest_path_dir = 0; // init dir with something random, this will never be
                                          // used if this node turns out to be the leaf node
            let mut snd_longest_path = 0;
            for &(child, weight) in adj_nodess[node]
                .iter()
                .filter(|&&(adj_node, _)| adj_node != parent)
            {
                let path_len =
                    weight + get_longest_path_down(child, node, adj_nodess, longest_2_paths_down);
                if path_len >= longest_path {
                    // previous longest is now second longest
                    snd_longest_path = longest_path;
                    // current path is now longest
                    longest_path = path_len;
                    longest_path_dir = child;
                } else if path_len >= snd_longest_path {
                    // current path isn't longest but is now second longest
                    snd_longest_path = path_len;
                }
            }
            longest_2_paths_down[node] = (longest_path, longest_path_dir, snd_longest_path);
            longest_path
        }
        let root = 0;
        let mut longest_2_paths_down = vec![(0, 0, 0); self.node_count()];
        get_longest_path_down(root, root, &self.adj_nodess, &mut longest_2_paths_down);

        fn get_longest_path(
            node: usize,
            parent: usize,
            parent_weight: i128,
            longest_2_paths_down: &[(i128, usize, i128)],
            longest_2_paths: &mut [(i128, usize, i128)],
        ) -> i128 {
            let (mut longest_path, mut longest_path_dir, mut snd_longest_path) =
                longest_2_paths_down[node];
            if node != parent {
                // not root => has parent
                // init parent as the longest path
                let (longest_parent_path, parent_path_dir, snd_longest_parent_path) =
                    longest_2_paths[parent]; // we expect the parent longest_2_paths
                                             // has already been populated
                let parent_path_len = if parent_path_dir != node {
                    // in another direction
                    longest_parent_path + parent_weight
                } else {
                    // take the len that's in another direction
                    snd_longest_parent_path + parent_weight
                };

                if parent_path_len >= longest_path {
                    // previous longest is now second longest
                    snd_longest_path = longest_path;
                    // parent path is now longest
                    longest_path = parent_path_len;
                    longest_path_dir = parent;
                } else if parent_path_len >= snd_longest_path {
                    // parent path isn't longest but is now second longest
                    snd_longest_path = parent_path_len;
                }
            }
            longest_2_paths[node] = (longest_path, longest_path_dir, snd_longest_path);
            longest_path
        }
        fn dfs(
            node: usize,
            parent: usize,
            parent_weight: i128,
            adj_nodess: &[Vec<(usize, i128)>],
            longest_2_paths_down: &[(i128, usize, i128)],
            longest_2_paths: &mut [(i128, usize, i128)],
            longest_paths: &mut [i128],
        ) {
            longest_paths[node] = get_longest_path(
                node,
                parent,
                parent_weight,
                longest_2_paths_down,
                longest_2_paths,
            );
            for &(child, weight) in adj_nodess[node]
                .iter()
                .filter(|&&(adj_node, _)| adj_node != parent)
            {
                dfs(
                    child,
                    node,
                    weight,
                    adj_nodess,
                    longest_2_paths_down,
                    longest_2_paths,
                    longest_paths,
                );
            }
        }
        let mut longest_2_paths = vec![(0, 0, 0); self.node_count()];
        let mut longest_paths = vec![0; self.node_count()];
        dfs(
            root,
            root,
            0,
            &self.adj_nodess,
            &longest_2_paths_down,
            &mut longest_2_paths,
            &mut longest_paths,
        );
        longest_paths
    }
}

#[cfg(test)]
mod test_tree {
    use super::*;

    // cphb 163
    const EDGES: &[(usize, usize, i128)] = &[
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 1),
        (2, 5, 1),
        (2, 6, 1),
        (6, 0, 3),
        (4, 7, 1),
    ];

    #[test]
    fn diameter() {
        let t = Tree::from_edges(EDGES.to_vec(), 8);
        assert_eq!(t.get_diameter(), 7);
    }

    #[test]
    fn all_longest_path_len() {
        let t = Tree::from_edges(EDGES.to_vec(), 8);
        assert_eq!(t.get_longest_path_lens(), vec![7, 5, 4, 6, 6, 5, 4, 7]);
    }
}

pub struct RootedTree {
    #[allow(dead_code)]
    root: usize,
    children: Vec<Vec<(usize, i128)>>,
    parents: SuccessorGraph,
    depths: Vec<usize>, // distance_to_parent, depth
}
impl RootedTree {
    pub fn new() -> Self {
        Self {
            root: 0,
            children: vec![Vec::new()],
            parents: SuccessorGraph::from_successors(vec![0]),
            depths: vec![0],
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        let mut children = Vec::with_capacity(capacity);
        children.push(Vec::new());
        let mut depths = Vec::with_capacity(capacity);
        depths.push(0);
        let mut parents = SuccessorGraph::with_capacity(capacity);
        parents.add_node(0);
        Self {
            root: 0,
            children,
            parents,
            depths,
        }
    }
    /// O(n)
    pub fn from_tree(tree: &Tree, root: usize) -> Self {
        let adj_nodess = tree.get_adj_nodess();
        let mut children = vec![Vec::new(); adj_nodess.len()];
        let mut parents = vec![0; adj_nodess.len()];
        let mut depths = vec![0; adj_nodess.len()];

        #[allow(clippy::too_many_arguments)]
        fn dfs(
            node: usize,
            parent: usize,
            dist_to_parent: i128,
            depth: usize,
            adj_nodess: &[Vec<(usize, i128)>],
            parents: &mut [usize],
            children: &mut [Vec<(usize, i128)>],
            depths: &mut [usize],
        ) {
            parents[node] = parent;
            children[parent].push((node, dist_to_parent));
            depths[node] = depth;
            for &(child, weight) in &adj_nodess[node] {
                if child != parent {
                    dfs(
                        child,
                        node,
                        weight,
                        depth + 1,
                        adj_nodess,
                        parents,
                        children,
                        depths,
                    );
                }
            }
        }
        dfs(
            root,
            root,
            0,
            0,
            adj_nodess,
            &mut parents,
            &mut children,
            &mut depths,
        );

        let mut parents = SuccessorGraph::from_successors(parents);
        parents.index_upto_kth_successor(*depths.iter().max().unwrap());

        Self {
            children,
            parents,
            depths,
            root,
        }
    }

    /// O(1)
    pub fn add_leaf(&mut self, parent: usize, dist_to_parent: i128) -> usize {
        let node = self.children.len();
        self.children[parent].push((node, dist_to_parent));
        self.parents.add_node(parent);
        let depth = self.depths[parent] + 1;
        self.depths.push(depth);
        self.parents.index_upto_kth_successor(depth);
        node
    }

    pub fn child(&self, node: usize) -> &Vec<(usize, i128)> {
        &self.children[node]
    }
    pub fn parent(&self, node: usize) -> Option<usize> {
        if self.depths[node] == 0 {
            None
        } else {
            Some(self.parents.get_successor(node))
        }
    }

    pub fn get_children(&self) -> &Vec<Vec<(usize, i128)>> {
        &self.children
    }

    /// O(log(n))
    pub fn lowest_common_ancestor(&self, node1: usize, node2: usize) -> usize {
        let Self {
            depths, parents, ..
        } = self;
        let depth = depths[node1].min(depths[node2]);
        let mut node1 = parents.get_kth_successor(node1, depths[node1] - depth);
        let mut node2 = parents.get_kth_successor(node2, depths[node2] - depth);
        let mut jump = pow2_ceil(depth);
        while jump > 0 {
            let p1 = parents.get_kth_successor(node1, jump);
            let p2 = parents.get_kth_successor(node2, jump);
            if p1 == p2 {
                jump /= 2;
            } else {
                node1 = p1;
                node2 = p2;
            }
        }
        self.parent(node1).unwrap_or(node1)
    }
}

#[cfg(test)]
mod test_rooted_tree {
    use super::*;

    // cphb 163
    const EDGES: &[(usize, usize, i128)] = &[
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 1),
        (2, 5, 1),
        (2, 6, 1),
        (6, 0, 3),
        (4, 7, 1),
    ];

    #[test]
    fn from_tree() {
        let t = RootedTree::from_tree(&Tree::from_edges(EDGES.to_vec(), 8), 1);
        assert_eq!(t.child(2), &vec![(5, 1), (6, 1)]);
        assert_eq!(t.parent(0), Some(6));
        assert_eq!(t.parent(2), Some(1));
        assert_eq!(t.parent(1), None);
    }

    #[test]
    fn modify() {
        let mut t = RootedTree::new();
        t.add_leaf(0, 1);
        assert_eq!(t.child(0), &vec![(1, 1)]);
        assert_eq!(t.parent(1), Some(0));
        assert_eq!(t.parent(0), None);
        assert_eq!(t.lowest_common_ancestor(0, 1), 0);
        let mut t = RootedTree::with_capacity(10);
        t.add_leaf(0, 1);
        assert_eq!(t.child(0), &vec![(1, 1)]);
        assert_eq!(t.parent(1), Some(0));
        assert_eq!(t.parent(0), None);
        assert_eq!(t.lowest_common_ancestor(0, 1), 0);
    }

    #[test]
    fn lowest_common_ancestor() {
        let t = RootedTree::from_tree(&Tree::from_edges(EDGES.to_vec(), 8), 1);
        assert_eq!(t.lowest_common_ancestor(0, 5), 2);
        assert_eq!(t.lowest_common_ancestor(1, 1), 1);
    }
}

pub struct SuccessorGraph {
    // kth successors of node x will be logkth_successors[log(k)][x]
    logkth_successors: Vec<Vec<usize>>,
}
impl SuccessorGraph {
    pub fn new() -> Self {
        Self {
            logkth_successors: vec![Vec::new()],
        }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            logkth_successors: vec![Vec::with_capacity(capacity)],
        }
    }
    pub fn from_successors(successors: Vec<usize>) -> Self {
        Self {
            logkth_successors: vec![successors],
        }
    }

    /// O(log(k))
    pub fn add_node(&mut self, successor: usize) -> usize {
        let node = self.logkth_successors[0].len();
        self.logkth_successors[0].push(successor);
        for logk in 1..self.logkth_successors.len() {
            let s = self.logkth_successors[logk - 1][node];
            let s2 = self.logkth_successors[logk - 1][s];
            self.logkth_successors[logk].push(s2);
        }
        node
    }

    /// O(nlog(k))
    pub fn index_upto_kth_successor(&mut self, k: usize) {
        let logkth_successors = &mut self.logkth_successors;
        while logkth_successors.len() < highest_one_bit(k) as usize {
            let n = logkth_successors.len();
            let lognth_successors = &logkth_successors[n - 1];
            let mut lognp1th_successors = Vec::with_capacity(lognth_successors.len());
            for i in 0..lognth_successors.len() {
                let s = lognth_successors[i];
                let ss = lognth_successors[s];
                lognp1th_successors.push(ss);
            }
            logkth_successors.push(lognp1th_successors);
        }
    }

    /// O(log(k))
    pub fn get_kth_successor(&self, node: usize, k: usize) -> usize {
        let logkth_successors = &self.logkth_successors;
        assert!(
            logkth_successors.len() >= highest_one_bit(k) as usize,
            "Need to index upto kthsuccessor first"
        );
        let mut node = node;
        let mut k = k;
        let mut logk = 0;
        while k > 0 {
            if k % 2 == 1 {
                node = logkth_successors[logk][node];
            }
            k /= 2;
            logk += 1;
        }
        node
    }

    pub fn get_successor(&self, node: usize) -> usize {
        self.logkth_successors[0][node]
    }

    /// O(n)
    pub fn get_cycle(&self, start: usize) -> Vec<usize> {
        let mut a = self.get_successor(start);
        let mut b = self.get_successor(self.get_successor(start));
        while a != b {
            a = self.get_successor(a);
            b = self.get_successor(b);
            b = self.get_successor(b);
        }
        // at this point, b is s away from f.
        a = start;
        while a != b {
            a = self.get_successor(a);
            b = self.get_successor(b);
        }
        let first = a;
        let mut node = self.get_successor(first);
        let mut cycle = vec![first];
        while node != first {
            cycle.push(node);
            node = self.get_successor(node);
        }
        cycle
    }
}

#[cfg(test)]
mod test_successor_graph {
    use super::*;

    #[test]
    fn add_node() {
        let mut g = SuccessorGraph::from_successors(vec![2, 4, 6, 5, 1, 1, 0]);
        g.index_upto_kth_successor(128);
        g.add_node(5);
        g.add_node(2);
        assert_eq!(g.get_kth_successor(7, 128), 1);
    }

    #[test]
    fn get_k_step_node_fn() {
        let mut g = SuccessorGraph::from_successors(vec![2, 4, 6, 5, 1, 1, 0, 5, 2]);
        g.index_upto_kth_successor(6);
        assert_eq!(g.get_kth_successor(3, 6), 1);
    }

    #[test]
    fn get_cycle() {
        let g = SuccessorGraph::from_successors(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 6]);
        assert_eq!(g.get_cycle(0), vec![6, 7, 8, 9]);
    }
}
