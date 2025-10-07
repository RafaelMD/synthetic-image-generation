import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional

from crewai.tools import tool
from PIL import Image

from .utils import analyze_context, generate_with_entity, judge_image


@dataclass
class ToolConfig:
    ai_client: Any
    entity: str
    context_limit: int


class ContextTools:
    def __init__(self, config: ToolConfig):
        self.config = config

    @tool("analyze_image_contexts")
    def analyze_image_contexts(self, image_path: str) -> str:
        contexts = analyze_context(
            self.config.ai_client,
            image_path,
            self.config.entity,
            self.config.context_limit,
        )
        return json.dumps(contexts, ensure_ascii=False)


class GenerationTools:
    def __init__(self, config: ToolConfig):
        self.config = config

    @tool("generate_entity_image")
    def generate_entity_image(
        self,
        image_path: str,
        context_option: Optional[str] = None,
    ) -> str:
        pil_image = generate_with_entity(
            self.config.ai_client,
            image_path,
            self.config.entity,
            context_option=context_option or None,
        )
        if pil_image is None:
            return json.dumps({"status": "error"})

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return json.dumps({"status": "success", "image_base64": encoded})


class JudgeTools:
    def __init__(self, config: ToolConfig):
        self.config = config

    @tool("judge_generated_image")
    def judge_generated_image(self, image_base64: str) -> str:
        try:
            raw = base64.b64decode(image_base64)
            pil_image = Image.open(BytesIO(raw))
        except Exception:
            return json.dumps({"status": False})

        judgement: Dict[str, bool] = judge_image(
            self.config.ai_client,
            pil_image,
            self.config.entity,
        )
        return json.dumps(judgement)
