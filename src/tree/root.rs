use std::collections::HashSet;

pub struct RootedTree {
    parents: Vec<Option<usize>>,
    childrens: Vec<Vec<usize>>,
}

pub fn root_tree(adj_nodes: &Vec<Vec<usize>>, root: usize) -> RootedTree {
    let mut parents = vec![None; adj_nodes.len()];
    let mut childrens = vec![Vec::new(); adj_nodes.len()];

    let mut to_visit = adj_nodes[root]
        .iter()
        .map(|&n| (root, n))
        .collect::<Vec<_>>();
    while let Some((parent, node)) = to_visit.pop() {
        parents[node] = Some(parent);
        childrens[parent].push(node);
        to_visit.extend(adj_nodes[node].iter().filter_map(|&n| {
            if n == parent {
                None
            } else {
                Some((parent, node))
            }
        }));
    }

    RootedTree { parents, childrens }
}

#[test]
fn test() {
    let edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 0), (4, 7)];
    let adj_nodes = super::adj_nodes::edges_to_adj_nodes(&edges);

    let rooted = root_tree(&adj_nodes, 1);

    assert_eq!(
        rooted
            .childrens
            .into_iter()
            .map(|nodes| HashSet::<usize>::from_iter(nodes.into_iter()))
            .collect::<Vec<_>>(),
        [
            vec![],
            vec![2usize, 3, 4],
            vec![5, 6],
            vec![],
            vec![7],
            vec![],
            vec![0],
            vec![]
        ]
        .into_iter()
        .map(|nodes| HashSet::<usize>::from_iter(nodes.into_iter()))
        .collect::<Vec<_>>(),
    );
    assert_eq!(rooted.parents[0], Some(6));
}
