import networkx as nx
def load_graph():
    g = nx.DiGraph(name="test")
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
    return g
