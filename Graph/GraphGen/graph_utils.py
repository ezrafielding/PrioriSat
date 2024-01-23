class DSU:
    def __init__(self):
        self.pa = {}
        self.sz = {}

    def size(self, x):
        if x not in self.sz:
            return 1
        else:
            return self.sz[x]

    def query(self, x):
        if x not in self.pa or x == self.pa[x]:
            return x
        else:
            self.pa[x] = self.query(self.pa[x])
            return self.pa[x]

    def merge(self, x, y):
        if x not in self.pa:
            self.pa[x] = x
            self.sz[x] = 1
        if y not in self.pa:
            self.pa[y] = y
            self.sz[y] = 1
        x = self.query(x)
        y = self.query(y)
        if x == y:
            return

        if self.sz[x] < self.sz[y]:
            x, y = y, x
        self.pa[y] = x
        self.sz[x] += self.sz[y]
        self.sz[y] = 0

def connected(graph):
    dsu = DSU()
    for i, j in graph.edges:
        dsu.merge(i, j)

    # assuming node 0 is always in graph
    if dsu.size(0) < len(graph.nodes):
        return False
    else:
        return True

def join(g1, g2):
    idx = {}
    start_idx = max(g1.nodes)+1

    for u in g2.nodes:
        idx[u] = start_idx
        g1.add_node(idx[u])
        start_idx += 1

    for u, v in g2.edges:
        g1.add_edge(idx[u], idx[v])

if __name__ == '__main__':
    import networkx as nx
    g = nx.gnm_random_graph(5, 7)
    assert connected(g)
    g = nx.gnm_random_graph(5, 3)
    assert not connected(g)

    g1 = nx.gnm_random_graph(5, 7)
    g2 = nx.gnm_random_graph(5, 7)
    join(g1, g2)
    print(g1.nodes, g1.edges)
    print(connected(g1))