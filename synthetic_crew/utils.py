import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Dict, Optional

import albumentations as A
import numpy as np
from google import genai
from google.genai.errors import ServerError
from PIL import Image


def build_genai_client(api_key: str) -> genai.Client:
    """Create a Google GenAI client using the provided API key."""
    return genai.Client(api_key=api_key)


def safe_json_extract(text: str, entity: str) -> Dict[str, str]:
    """Extract a JSON object from a string, falling back to a default description."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"1": f"{entity} in the scene (fallback)"}


def analyze_context(
    ai_client: genai.Client,
    image_path: str,
    entity: str,
    context_number: int,
) -> Dict[str, str]:
    """Use Gemini to analyse an image and propose contexts for the entity."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt_text = (
        f"Analyze this image and return possible scenarios where the entity '{entity}' could be placed. "
        f"The output must be ONLY a valid JSON object with keys as integers and values as short English descriptions. "
        f"Example: {{\"1\": \"{entity} standing in the roadside\", \"2\": \"{entity} standing in the middle of the road\"}}. "
        f"Limit yourself to a maximum of {context_number} values. Only valid JSON."
    )

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}},
    ]

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text_out = response.candidates[0].content.parts[0].text
    return safe_json_extract(text_out, entity)


def generate_with_entity(
    ai_client: genai.Client,
    image_path: str,
    entity: str,
    context_option: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Image.Image]:
    """Call Gemini to generate an augmented image with the requested entity."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt_text = (
        f"Add {entity} in this context: {context_option}."
        if context_option
        else f"Add {entity} into the scene."
    )

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": mime_type, "data": base64_image}},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=prompt,
            )
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return Image.open(BytesIO(part.inline_data.data))
        except ServerError:
            if attempt < max_retries:
                time.sleep(3)
            else:
                return None
    return None


def judge_image(ai_client: genai.Client, pil_image: Image.Image, entity: str) -> Dict[str, bool]:
    """Ask Gemini to judge whether the generated entity looks realistic."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt_text = (
        f"You are a strict evaluator of AI-generated content. "
        f"Look ONLY at the entity '{entity}' in the image. "
        f"If the entity looks artificial, fake, poorly blended, distorted, or clearly AI-generated, "
        f"respond with this exact JSON: {{\"status\": false}}. "
        f"If the entity looks natural enough in the context of the scene (even if not perfect), "
        f"respond with this exact JSON: {{\"status\": true}}. "
        f"Do not include explanations, only the JSON."
    )

    prompt = [
        {"text": prompt_text},
        {"inlineData": {"mimeType": "image/png", "data": base64_image}},
    ]

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text_out = response.candidates[0].content.parts[0].text.strip()
    try:
        return json.loads(text_out)
    except Exception:
        return {"status": False}


TRANSFORM = A.Compose([
    A.HorizontalFlip(p=1),
])


def augment_image(pil_image: Image.Image) -> Image.Image:
    """Apply deterministic data augmentation to the generated image."""
    img = np.array(pil_image)
    augmented = TRANSFORM(image=img)["image"]
    return Image.fromarray(augmented)
