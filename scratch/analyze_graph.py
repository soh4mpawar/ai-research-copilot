import sys
import os
sys.path.insert(0, os.path.abspath("."))

from backend.graph.neo4j_builder import CitationGraphEngine

engine = CitationGraphEngine()
g = engine.graph
total_nodes = g.number_of_nodes()
total_edges = g.number_of_edges()
connected_nodes = [n for n in g.nodes() if g.degree(n) > 0]
isolated_nodes = [n for n in g.nodes() if g.degree(n) == 0]

print(f"Total Nodes: {total_nodes}")
print(f"Total Edges: {total_edges}")
print(f"Connected Nodes (degree > 0): {len(connected_nodes)}")
print(f"Isolated Nodes (degree == 0): {len(isolated_nodes)}")

# Degree distribution of top hub nodes:
degrees = sorted([(n, g.degree(n), g.in_degree(n), g.out_degree(n), g.nodes[n].get('label', '')) for n in connected_nodes], key=lambda x: x[1], reverse=True)
print("\nTop 10 Hub Papers by degree:")
for d in degrees[:10]:
    print(f"  [{d[0]}] Total Deg: {d[1]} (In: {d[2]}, Out: {d[3]}) — {d[4][:40]}")
