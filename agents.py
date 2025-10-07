from crewai import Agent
from tools import (
    ImageContextAnalyzerTool,
    SyntheticImageGeneratorTool,
    ImageQualityJudgeTool,
    DataAugmentationTool,
    ReportGeneratorTool
)

def create_image_analyzer_agent():
    """
    Image Analyzer Agent - Context Analysis Specialist
    Analyzes input images and identifies optimal placement scenarios for entities
    """
    return Agent(
        role="Context Analysis Specialist",
        goal="Analyze input images and identify optimal placement scenarios for entities to create realistic synthetic images",
        backstory=(
            "You are an expert in visual scene understanding and spatial analysis. "
            "You specialize in identifying realistic positions and contexts where objects "
            "can be naturally inserted into scenes. Your keen eye for detail ensures that "
            "every placement suggestion maintains visual coherence and realism."
        ),
        tools=[ImageContextAnalyzerTool()],
        verbose=True,
        allow_delegation=False
    )

def create_image_generator_agent():
    """
    Image Generator Agent - Synthetic Image Creation Specialist
    Generates high-quality synthetic images by inserting entities into analyzed contexts
    """
    return Agent(
        role="Synthetic Image Creation Specialist",
        goal="Generate high-quality synthetic images by seamlessly inserting entities into provided contexts",
        backstory=(
            "You are a master of AI-powered image manipulation with years of experience "
            "in digital content creation. You specialize in seamlessly blending entities "
            "into existing scenes, ensuring natural lighting, perspective, and integration. "
            "You handle API challenges gracefully and never give up until the perfect image is created."
        ),
        tools=[SyntheticImageGeneratorTool()],
        verbose=True,
        allow_delegation=False
    )

def create_quality_judge_agent():
    """
    Quality Judge Agent - Quality Assurance Evaluator
    Ensures only realistic, high-quality synthetic images are accepted
    """
    return Agent(
        role="Quality Assurance Evaluator",
        goal="Ensure only realistic, high-quality synthetic images pass quality control by identifying and rejecting artificial-looking results",
        backstory=(
            "You are a strict evaluator with an exceptionally keen eye for artificial artifacts. "
            "Years of experience in quality assurance have made you an expert at spotting "
            "poor blending, distortions, and unnatural elements. You maintain the highest "
            "standards and only approve images that could pass as authentic photographs."
        ),
        tools=[ImageQualityJudgeTool(), DataAugmentationTool(), ReportGeneratorTool()],
        verbose=True,
        allow_delegation=False
    )
