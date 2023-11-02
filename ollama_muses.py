import os
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import networkx as nx
import json
import networkx.drawing.nx_agraph as nx_agraph
from pyvis.network import Network
from langchain.schema import StrOutputParser

llm = Ollama(
    model="mistral",
    #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])    
)

def dollm(text):
    prompt = PromptTemplate.from_template(
        """Create a poetic description of a journey through the celestial symphony guided by emojis and symbols. This Involves infusing emojis with wisdom, invoking the Muses' presence, constructing emoji thought patterns, shaping a tapestry, having emojis converse, embarking on a cosmic journey, dancing of expression, unveiling the emoji tapestry, chronicling the journey, and sharing cosmic harmony. Now with this inspiration, interpret the following: {text}. Now rewrite it with emojis and feeling!"""    )
    runnable = prompt | llm | StrOutputParser()
    data = runnable.invoke({"text": text})
    print(data)
    return data


size = 50
for index in range(size):
    name= f'Component{index}'
    name2= f'StrongComponent{index}'
    for fname in (name, name2):
        print(fname)
        ofn = f"graphs/graph{fname}_emoji.txt"
        if not os.path.exists(ofn):
            with open(f"graphs/graph{fname}.txt", "r") as inf:
                data = inf.read()
                print(data)
                with open(f"graphs/graph{fname}_emoji.txt", "w") as of:
                    data2 = dollm(data)
                    of.write(data2)
