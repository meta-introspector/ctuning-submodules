# Copyright James Michael Dupont 2023

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import networkx as nx
import json
import networkx.drawing.nx_agraph as nx_agraph
from pyvis.network import Network
from langchain.schema import StrOutputParser
import load_graph
import numpy as np
import networkx as nx

llm = Ollama( model="mistral")

def dollm(text):
    prompt = PromptTemplate.from_template(
        """Given this starting ontology @prefix : <http://ctuning.org/ml-benchmark-ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://ctuning.org/ml-benchmark-ontology> a owl:Ontology .

:AutomatedDesignSpaceExploration a owl:Class ;
    rdfs:subClassOf :Exploration .

:Standardization a owl:Class ;
    rdfs:subClassOf :Process .

:Workflow a owl:Class ;
    rdfs:subClassOf :Process .

:hasBenchmark a owl:ObjectProperty ;
    rdfs:domain :Model ;
    rdfs:range :Benchmark .

:mlperfInferencev1.0 a :MLPerfInference,
        owl:NamedIndividual .

:reproducibilityReportMLPerfInferencev1.1 a :ReproducibilityReport,
        owl:NamedIndividual .

:Exploration a owl:Class ;
    rdfs:subClassOf :Analysis .

:MLPerfInference a owl:Class ;
    rdfs:subClassOf :Benchmark .

:Report a owl:Class ;
    rdfs:subClassOf :Documentation .

:ReproducibilityReport a owl:Class ;
    rdfs:subClassOf :Report .

:Benchmark a owl:Class ;
    rdfs:subClassOf :Evaluation .
 Interpret this following nodes extracted from headings of readmes from ctuning.org : {text} Now please create further turtle owl statements and relationships for the model."""
    )
    runnable = prompt | llm | StrOutputParser()
    print(text)
    data = runnable.invoke({"text": text})
    print(data)
    return data


g = load_graph.load_graph()
forest = nx.minimum_spanning_tree(g.to_undirected())

group = []
count = 0
for f in forest:
    #print(f)
    group.append(f)
    count = count + 1
    
    if len(group)> 40:
        #print(group)

        name = "group_{count}"
        count = count + 1
        data = dollm(str(group))
        with open(f"graphs/graph{name}.txt", "w") as f:
            f.write(str(group))
            f.write(data)
        group =[]
