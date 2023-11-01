import networkx as nx
import json
import networkx.drawing.nx_agraph as nx_agraph
from pyvis.network import Network
#from networkx.drawing.nx_pydot import write_dot


g = nx.DiGraph(name="test")

data="""write python script
read in lines of outline, has : separating filename from content
split out filename, number of stars and rest of text
if filename changes break
look at structure, if goes in up (less stars), break
if goes down more stars than before, capture pairs
"""
last_filename = ""
last_stars    = 0
last_content  = "start"

lines = []
with open("data/outline.txt") as fi:
    for l in fi:
        l = l.strip()
        parts = l.split(":")
        filename=parts[0]
        rest = ":".join(parts[1:])
        
        parts = rest.split(" ")
        stars=parts[0]
        #lines.append(l)

        content = " ".join(parts[1:]).replace(":","_").replace(";","_").replace("\n","_").replace("\t","_").replace(" ","_")

        changed= 0
        if filename != last_filename :
            changed = 0
        elif len(stars) < last_stars :
            changed = 1
        if changed:
            #print("|".join([str(last_stars), str(len(stars)),  last_content, content]))
            
            if (last_content, content) in g.edges():
                data = g.get_edge_data(last_content, content)
                g.add_edge(last_content, content,  weight=data['weight']+1)
            else:
                g.add_edge(last_content, content,  weight=1)                


        last_filename = filename
        last_stars    = len(stars)
        last_content  = content

nx.write_gml(g,"graphs/graph.gml")
nx.write_adjlist(g, "graphs/graph.adjlist")
nx.write_multiline_adjlist(g, "graphs/graph.adjlist2")
nx.write_weighted_edgelist(g, "graphs/graph.edgelist")
#pos = nx_agraph.graphviz_layout(g)
#pos = nx.nx_agraph.graphviz_layout(G)
#nx.draw(G, pos=pos)
#write_dot(G, 'file.dot')

nt = Network(notebook=True, width="100%", height="800px",
             #directed=True
             )
for edge in g.edges():
    source_id_str = edge[0]
    target_id_str = edge[1]
    edge_id_str = (
        f"{source_id_str}_to_{target_id_str}"  # Construct a unique edge id
    )
    nt.add_node(source_id_str)
    nt.add_node(target_id_str)
    nt.add_edge(source_id_str, target_id_str, id=edge_id_str)
hierarchical_options = {
        "enabled": True,
        #"levelSeparation": 200,  # Increased vertical spacing between levels
        #"nodeSpacing": 250,  # Increased spacing between nodes on the same level
        #"treeSpacing": 250,  # Increased spacing between different trees (for forest)
        "blockShifting": True,
        # "edgeMinimization": True,
        "parentCentralization": True,
        #"direction": "UD",
        "sortMethod": "directed",
    }
physics_options = {
        "stabilization": {
            "enabled": True,
            "iterations": 1000,  # Default is often around 100
        },
        "hierarchicalRepulsion": {
            "centralGravity": 0.0,
                "springLength": 700,  # Increased edge length
                "springConstant": 0.01,
                "nodeDistance": 750,  # Increased minimum distance between nodes
                "damping": 0.09,
            },
            "solver": "hierarchicalRepulsion",
            "timestep": 0.5,
        }
nt.options = {
            "nodes": {
                "font": {
                    "size": 12,  # Increased font size for labels
                    "color": "black",  # Set a readable font color
                },
                "shapeProperties": {"useBorderWithImage": True},
            },
            "edges": {
                "length": 1050,  # Increased edge length
            },
            "physics": physics_options,
            "layout": {"hierarchical": hierarchical_options},
        }
graph_data = {"nodes": nt.nodes, "edges": nt.edges}
json_graph = json.dumps(graph_data)
with open(f"graphs/graph.json", "w") as f:
    f.write(json_graph)
    nt.show(f"graphs/graph.html")

if True:
    size = 30
    dag = g.to_undirected()
    #largest_components = sorted(nx.strongly_connected_components(dag), key=len, reverse=True)[:size]
    largest_components = sorted(nx.connected_components(dag), key=len, reverse=True)[:size]
    #print(len(largest_components))
    for index,comp in enumerate(largest_components):
        name= f'Component{index}'
        component=dag.subgraph(comp)
        nt = Network(notebook=True, width="100%", height="800px",
                     #directed=True
                     )
        for edge in component.edges():
            source_id_str = edge[0]
            target_id_str = edge[1]
            edge_id_str = (
                f"{source_id_str}_to_{target_id_str}"  # Construct a unique edge id
            )
            nt.add_node(source_id_str)
            nt.add_node(target_id_str)
            nt.add_edge(source_id_str, target_id_str, id=edge_id_str)
        hierarchical_options = {
            "enabled": True,
            #"levelSeparation": 200,  # Increased vertical spacing between levels
            #"nodeSpacing": 250,  # Increased spacing between nodes on the same level
            #"treeSpacing": 250,  # Increased spacing between different trees (for forest)
            "blockShifting": True,
           # "edgeMinimization": True,
            "parentCentralization": True,
            #"direction": "UD",
            "sortMethod": "directed",
        }
        physics_options = {
            "stabilization": {
                "enabled": True,
                "iterations": 1000,  # Default is often around 100
            },
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 700,  # Increased edge length
                "springConstant": 0.01,
                "nodeDistance": 750,  # Increased minimum distance between nodes
                "damping": 0.09,
            },
            "solver": "hierarchicalRepulsion",
            "timestep": 0.5,
        }
        nt.options = {
            "nodes": {
                "font": {
                    "size": 12,  # Increased font size for labels
                    "color": "black",  # Set a readable font color
                },
                "shapeProperties": {"useBorderWithImage": True},
            },
            "edges": {
                "length": 1050,  # Increased edge length
            },
            "physics": physics_options,
            "layout": {"hierarchical": hierarchical_options},
        }
        graph_data = {"nodes": nt.nodes, "edges": nt.edges}
        json_graph = json.dumps(graph_data)
        with open(f"graphs/graph{name}.json", "w") as f:
            f.write(json_graph)
        nt.show(f"graphs/graph{name}.html")
