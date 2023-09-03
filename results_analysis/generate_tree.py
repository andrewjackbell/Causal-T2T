
from os import path
import networkx as nx
import pandas
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import argparse

def generate_tree(result_dir,result_name):
    with open(path.join(result_dir,result_name,"causal_tree.csv")) as f:
        tree_df = pandas.read_csv(f)

    graph = nx.DiGraph()

    nodes = []
    for index,row in tree_df.iterrows():
        level = row['level']
        z_score = row['z score']
        route = row['route']

        nodes.append((level,z_score,route))

    nodes.sort(key=lambda x:(x[0],x[1]))

    used_counts = [0,0,0]
    used_terms = set()

    for index,row in tree_df.iterrows():
        level = row['level']
        route = str(row['route']).lower()
        value = row['value'].lower()
        z_score = row['z score']

        if 'hepatic' in value or 'po' in value:
            continue    

        if value in used_terms:
            continue

        if used_counts[level]>level:
            continue
            
        if level==0:
            node_name=value
            graph.add_edge(node_name,"analgesics-induced liver failure", weight=z_score)
            used_counts[level]+=1
            used_terms.add(value)   
        else:
            node_name = str(route)+f"->{value}"
            if graph.has_node(route):
                graph.add_edge(node_name,route,weight=z_score)
                used_counts[level]+=1
                used_terms.add(value)
            
        
    plt.switch_backend('agg')
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    plt.figure(figsize=(20, 16))
    nx.draw_networkx_nodes(graph, pos,node_size=1000)
    nx.draw_networkx_edges(graph, pos,arrowsize=40)

    edge_labels = {(u, v): f"{d['weight']:0.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, font_size=22,edge_labels=edge_labels)

    node_labels = {node: node.split('->')[-1] for node in graph.nodes()}
    for node, label in node_labels.items():
        x, y = pos[node]
        plt.text(x, y + 8, label, ha="center", fontsize=24)
        items = nx.get_edge_attributes(graph,'weight').items()
        edge_labels = {i:f"{j:0f}" for i,j in items}

    plt.savefig(result_name+'.png',dpi=300)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--result-name', type=str)
    args = parser.parse_args()

    generate_tree(args.result_dir,args.result_name)

if __name__ == "__main__":
    main()