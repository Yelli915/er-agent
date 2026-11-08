"""Construct the local graph dependencies without creating external API clients."""

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore

from er_finder.agent.adapters import VisitSelection
from er_finder.agent.execution import ModelCounter
from er_finder.agent.graph import build_graph
from er_finder.guardrails.contracts import Classifier
from er_finder.guardrails.triage import InputClassifier
from er_finder.memory.context import ERFinderContext, TransportMode
from er_finder.memory.session import SessionManager
from er_finder.memory.store import ERFinderStore
from er_finder.search.contracts import SearchProvider
from er_finder.search.service import SearchSession


@dataclass(frozen=True)
class RunnerDependencies:
    context: ERFinderContext
    session: SearchSession
    store: Any
    profiles: ERFinderStore
    visit: VisitSelection
    checkpointer: Any
    session_manager: SessionManager
    classifier: Classifier
    counter: ModelCounter
    graph: Any
    run_config: RunnableConfig


def build_runner_dependencies(
    provider: SearchProvider,
    model: Any,
    *,
    classifier: Classifier | None,
    user_id: str,
    transport: TransportMode,
    store: Any,
    checkpointer: Any,
    server_planned_calls: bool,
    deterministic_reply: bool,
) -> RunnerDependencies:
    context = ERFinderContext(user_id, transport)
    session = SearchSession(provider, transport)
    store = store if store is not None else InMemoryStore()
    profiles = ERFinderStore(store, user_id)
    visit = VisitSelection(profiles, session)
    checkpointer = (
        checkpointer
        if checkpointer is not None
        else InMemorySaver(
            serde=JsonPlusSerializer(
                # Nested candidates are data; only the reply is revived.
                allowed_msgpack_modules=[("er_finder.models", "ERSearchReply")],
            )
        )
    )
    session_manager = SessionManager(checkpointer)
    classifier = classifier if classifier is not None else InputClassifier()
    counter = ModelCounter()
    graph = build_graph(
        model,
        session=session,
        profiles=profiles,
        visit=visit,
        counter=counter,
        store=store,
        checkpointer=checkpointer,
        server_planned_calls=server_planned_calls,
        deterministic_reply=deterministic_reply,
    )
    run_config = session_manager.build_config(user_id)
    run_config["configurable"].update(session=session, visit=visit)
    return RunnerDependencies(
        context,
        session,
        store,
        profiles,
        visit,
        checkpointer,
        session_manager,
        classifier,
        counter,
        graph,
        run_config,
    )
