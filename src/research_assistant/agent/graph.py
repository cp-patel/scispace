"""Wire the nodes into a LangGraph graph.

plan -> gather -> rank -> screen -> select -> approve -> ingest -> outline
  -> one 'section' subgraph per section, in parallel (Send)
       retrieve -> compress -> write -> verify -> (revise -> verify, at most 2 rounds) -> finish
  -> assemble
"""

from langgraph.graph import END, START, StateGraph

from research_assistant.agent.nodes import JobNodes
from research_assistant.agent.state import JobState, SectionOut, SectionState

LEVEL_ONE = ("plan", "gather", "rank", "screen", "select", "approve", "ingest", "outline")


def build_graph(nodes: JobNodes, checkpointer=None):
    section = StateGraph(SectionState, output_schema=SectionOut)
    for name in ("retrieve", "compress", "write", "verify", "revise", "finish"):
        section.add_node(name, getattr(nodes, name))
    section.add_edge(START, "retrieve")
    section.add_edge("retrieve", "compress")
    section.add_edge("compress", "write")
    section.add_edge("write", "verify")
    section.add_conditional_edges("verify", nodes.after_verify, ["revise", "finish"])
    section.add_edge("revise", "verify")
    section.add_edge("finish", END)

    graph = StateGraph(JobState)
    for name in LEVEL_ONE + ("assemble",):
        graph.add_node(name, getattr(nodes, name))
    graph.add_node("section", section.compile())
    graph.add_edge(START, LEVEL_ONE[0])
    for a, b in zip(LEVEL_ONE, LEVEL_ONE[1:]):
        graph.add_edge(a, b)
    graph.add_conditional_edges("outline", nodes.fan_out, ["section"])
    graph.add_edge("section", "assemble")
    graph.add_edge("assemble", END)
    return graph.compile(checkpointer=checkpointer)
