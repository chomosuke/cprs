use std::{cmp::max, collections::HashSet};

pub fn edges_to_adj_nodes(edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj_nodes = Vec::new();
    for &(u, v) in edges {
        let max_node = max(u, v);
        while max_node >= adj_nodes.len() {
            adj_nodes.push(Vec::new());
        }
        adj_nodes[u].push(v);
        adj_nodes[v].push(u);
    }
    adj_nodes
}

#[test]
fn test() {
    let edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 0), (4, 7)];

    let adj_nodes = edges_to_adj_nodes(&edges);
    for (u, v) in edges {
        assert!(adj_nodes[u].contains(&v));
        assert!(adj_nodes[v].contains(&u));
    }
    for adj_node in &adj_nodes {
        assert_eq!(
            HashSet::<usize>::from_iter(adj_node.iter().cloned()).len(),
            adj_node.len(),
        );
    }
    assert!(adj_nodes.len() == 8)
}
