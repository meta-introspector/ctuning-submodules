import networkx as nx
import networkx.drawing.nx_agraph as nx_agraph
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
with open("outline.txt") as fi:
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

nx.write_gml(g,"graph.gml")
nx.write_adjlist(g, "graph.adjlist")
nx.write_multiline_adjlist(g, "graph.adjlist2")
nx.write_weighted_edgelist(g, "graph.edeelist")
#pos = nx_agraph.graphviz_layout(g)
#pos = nx.nx_agraph.graphviz_layout(G)
#nx.draw(G, pos=pos)
#write_dot(G, 'file.dot')
