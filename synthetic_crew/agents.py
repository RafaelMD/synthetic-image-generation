from typing import Callable, List

from crewai import Agent


def create_context_agent(entity: str, tools: List[Callable]) -> Agent:
    return Agent(
        role="Image Context Analyst",
        goal=f"Understand background scenes and suggest realistic contexts to insert {entity}.",
        backstory=(
            "You specialise in visual scene analysis. "
            "Your job is to read an image and describe a handful of grounded, concise contexts "
            "where the requested entity could fit believably."
        ),
        allow_delegation=False,
        verbose=False,
        tools=tools,
    )


def create_generation_agent(entity: str, tools: List[Callable]) -> Agent:
    return Agent(
        role="Entity Image Generator",
        goal=(
            f"Insert the entity {entity} into background images while keeping lighting and perspective coherent."
        ),
        backstory=(
            "You are a creative yet precise image editor who works with AI models. "
            "You are responsible for calling the generation tool with the right context instructions "
            "and returning the generated image encoded as base64."
        ),
        allow_delegation=False,
        verbose=False,
        tools=tools,
    )


def create_judge_agent(entity: str, tools: List[Callable]) -> Agent:
    return Agent(
        role="Entity Quality Judge",
        goal=(
            f"Examine the generated images and confirm if the inserted {entity} looks realistic and well blended."
        ),
        backstory=(
            "You are a strict reviewer focused on visual authenticity. "
            "You only return whether the entity passes inspection, encoded as JSON."
        ),
        allow_delegation=False,
        verbose=False,
        tools=tools,
    )
