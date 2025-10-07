from crewai import Task


def create_context_task(agent: "Agent") -> Task:
    return Task(
        name="Context Discovery",
        description=(
            "Analyze the image located at {image_path} and propose up to {context_limit} "
            "plausible short contexts where the entity {entity} could be inserted. "
            "Always call the analyze_image_contexts tool and return only the JSON it produces."
        ),
        expected_output=(
            "A JSON object encoded as string with integer keys and short descriptions of contexts."
        ),
        agent=agent,
    )


def create_generation_task(agent: "Agent") -> Task:
    return Task(
        name="Entity Synthesis",
        description=(
            "Using the contexts provided previously ({context_suggestions}), "
            "choose one context id at a time and call the generate_entity_image tool with the "
            "image path {image_path} and the context description. "
            "Return the JSON response from the tool."
        ),
        expected_output=(
            "JSON dictionary describing the status of the generation and the image encoded in base64."
        ),
        agent=agent,
    )


def create_judge_task(agent: "Agent") -> Task:
    return Task(
        name="Quality Review",
        description=(
            "Inspect the generated image encoded in base64 ({generated_image_base64}). "
            "Call the judge_generated_image tool to obtain the quality decision and return the JSON response."
        ),
        expected_output="JSON object with a boolean status field.",
        agent=agent,
    )
