"""
This module defines the PocketFlow Flows for the Research Assistant.
Flows orchestrate the execution of nodes to perform complex tasks like
the initial paper search and interactive refinement sessions.
"""

from __future__ import annotations
from pocketflow import Flow
from .nodes import (
    InitialQueryNode,
    InteractiveClarificationNode,
    KeywordExtractionNode,
    KeywordExpansionNode,
    BatchArxivNode,
    EmbeddingFilterNode,
    LLMAnalysisNode,
    LLMFilterNode,
    SaveResultsNode,
    LoadProjectNode,
    PromptUserNode,
    InterpretQueryNode,
    RescoreNode,
    MetaAnalysisNode,
    UpdateProjectNode,
)


def create_research_search_flow() -> Flow:
    """
    Creates the PocketFlow for the initial massive-scale paper search.

    This flow takes a user query, searches arXiv, filters results by
    embedding similarity, performs a deeper analysis with an LLM, and
    initializes a new research project with the findings.

    Returns:
        A PocketFlow `Flow` object ready to be executed.
    """
    # Instantiate all nodes for the search workflow
    batch_arxiv_node = BatchArxivNode()
    embedding_filter_node = EmbeddingFilterNode()
    llm_analysis_node = LLMAnalysisNode()
    save_results_node = SaveResultsNode()

    # Connect the nodes sequentially using the >> operator for a clean look
    (
        batch_arxiv_node
        >> embedding_filter_node
        >> llm_analysis_node
        >> save_results_node
    )

    # Create the flow and set its starting point
    flow = Flow(start=batch_arxiv_node)

    return flow


def create_interactive_refinement_flow() -> Flow:
    """
    Creates the PocketFlow for the interactive refinement session.

    This flow loads an existing project, enters a loop to get user input,
    re-scores papers, generates meta-analysis, and saves new versions
    of the results.

    Returns:
        A PocketFlow `Flow` object for the refinement loop.
    """
    # Instantiate nodes for the refinement workflow
    load_project_node = LoadProjectNode()
    prompt_user_node = PromptUserNode()
    interpret_query_node = InterpretQueryNode()
    rescore_node = RescoreNode()
    meta_analysis_node = MetaAnalysisNode()
    update_project_node = UpdateProjectNode()

    # Define the refinement cycle using the >> operator
    (
        prompt_user_node
        >> interpret_query_node
        >> rescore_node
        >> meta_analysis_node
        >> update_project_node
    )

    # The full flow starts by loading the project, then runs the cycle.
    # The looping logic will be handled by the CLI runner.
    main_flow = Flow()
    load_project_node >> prompt_user_node  # Connect loading to the start of the cycle
    main_flow.start(load_project_node)

    return main_flow


def create_interactive_search_flow() -> Flow:
    """
    Creates and connects all the nodes for the main interactive search workflow.
    """
    # Instantiate nodes
    initial_query_node = InitialQueryNode()
    interactive_clarification_node = InteractiveClarificationNode()
    keyword_extraction_node = KeywordExtractionNode()
    keyword_expansion_node = KeywordExpansionNode()
    batch_arxiv_node = BatchArxivNode()
    embedding_filter_node = EmbeddingFilterNode()
    llm_analysis_node = LLMAnalysisNode()
    llm_filter_node = LLMFilterNode()
    save_results_node = SaveResultsNode()

    # Define the flow
    (
        initial_query_node
        >> interactive_clarification_node
        >> keyword_extraction_node
        >> keyword_expansion_node
        >> batch_arxiv_node
        >> embedding_filter_node
        >> llm_analysis_node
        >> llm_filter_node
        >> save_results_node
    )

    # Create the flow and set its starting point
    flow = Flow(start=initial_query_node)

    return flow
