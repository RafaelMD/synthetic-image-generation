import base64
import json
import os
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Optional

from crewai import Crew, Process
from PIL import Image
from tqdm import tqdm

from .agents import create_context_agent, create_generation_agent, create_judge_agent
from .tasks import create_context_task, create_generation_task, create_judge_task
from .tools import ContextTools, GenerationTools, JudgeTools, ToolConfig
from .utils import augment_image, build_genai_client


@dataclass
class CrewResult:
    success: bool
    payload: Optional[str] = None
    error: Optional[str] = None


class SyntheticImageGenerationCrew:
    """End-to-end pipeline orchestrated by CrewAI agents."""

    def __init__(
        self,
        api_key: str,
        entity: str,
        context_limit: int,
        input_folder: str,
        output_folder: str,
        discard_folder: str,
    ) -> None:
        self.api_key = api_key
        self.entity = entity
        self.context_limit = context_limit
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.discard_folder = discard_folder

        self.ai_client = build_genai_client(api_key)
        config = ToolConfig(
            ai_client=self.ai_client,
            entity=entity,
            context_limit=context_limit,
        )
        self.context_tools = ContextTools(config)
        self.generation_tools = GenerationTools(config)
        self.judge_tools = JudgeTools(config)
        self.crewai_enabled = True

    # ------------------------------------------------------------------
    # Crew builders
    def _build_context_crew(self) -> Crew:
        agent = create_context_agent(
            self.entity,
            tools=[self.context_tools.analyze_image_contexts],
        )
        task = create_context_task(agent)
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
        )

    def _build_generation_crew(self, context_suggestions: Dict[str, str]) -> Crew:
        agent = create_generation_agent(
            self.entity,
            tools=[self.generation_tools.generate_entity_image],
        )
        task = create_generation_task(agent)
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
        )

    def _build_judge_crew(self, image_base64: str) -> Crew:
        agent = create_judge_agent(
            self.entity,
            tools=[self.judge_tools.judge_generated_image],
        )
        task = create_judge_task(agent)
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
        )

    # ------------------------------------------------------------------
    def _run_with_crewai(self, crew_builder, inputs: Dict[str, str], fallback_callable):
        if not self.crewai_enabled:
            return CrewResult(success=True, payload=fallback_callable())

        try:
            crew: Crew = crew_builder()
            result = crew.kickoff(inputs=inputs)
            parsed = self._extract_output(result)
            if parsed is None:
                raise ValueError("Empty crew output")
            return CrewResult(success=True, payload=parsed)
        except Exception as exc:  # noqa: BLE001
            # Disable CrewAI execution for the rest of the run to avoid repeated failures
            self.crewai_enabled = False
            return CrewResult(success=True, payload=fallback_callable(), error=str(exc))

    @staticmethod
    def _extract_output(result) -> Optional[str]:  # noqa: ANN001
        if result is None:
            return None
        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "result") and result.result:
            return str(result.result)
        if hasattr(result, "raw") and result.raw:
            return str(result.raw)
        if hasattr(result, "outputs") and result.outputs:
            # Some CrewAI versions expose a list of outputs
            output = result.outputs[-1]
            if isinstance(output, str):
                return output
            if isinstance(output, dict) and "output" in output:
                return str(output["output"])
        if hasattr(result, "tasks_output") and result.tasks_output:
            last = result.tasks_output[-1]
            if isinstance(last, dict) and "output" in last:
                return str(last["output"])
        # Fallback to string conversion
        return str(result)

    # ------------------------------------------------------------------
    def prepare_folders(self) -> None:
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.discard_folder, exist_ok=True)

        entity_output = self.entity_output_folder
        os.makedirs(entity_output, exist_ok=True)

    @property
    def entity_output_folder(self) -> str:
        return os.path.join(self.output_folder, self.entity)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        self.prepare_folders()

        report = {
            "entity": self.entity,
            "total_images": 0,
            "api_success": 0,
            "api_failures": 0,
            "augmented_images": 0,
            "discarded": 0,
            "contexts": {},
        }

        start_time = time.time()
        for img_file in tqdm(os.listdir(self.input_folder)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            report["total_images"] += 1
            input_path = os.path.join(self.input_folder, img_file)

            contexts = self._obtain_contexts(input_path)
            report["contexts"][img_file] = contexts

            for idx, context_option in contexts.items():
                generation_outcome = self._generate_image(input_path, context_option, contexts)
                if not generation_outcome:
                    report["api_failures"] += 1
                    continue

                report["api_success"] += 1
                status, pil_image = generation_outcome
                if not status:
                    self._save_discard(img_file, idx, pil_image)
                    report["discarded"] += 1
                    continue

                self._persist_outputs(img_file, idx, pil_image, report)

        elapsed = time.time() - start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        report["processing_time"] = f"{int(h)}h {int(m)}m {int(s)}s"

        report_path = os.path.join(self.entity_output_folder, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    # ------------------------------------------------------------------
    def _obtain_contexts(self, image_path: str) -> Dict[str, str]:
        fallback = lambda: self.context_tools.analyze_image_contexts(image_path)  # noqa: E731
        result = self._run_with_crewai(
            self._build_context_crew,
            {"image_path": image_path, "entity": self.entity, "context_limit": str(self.context_limit)},
            fallback,
        )
        try:
            return json.loads(result.payload or "{}")
        except json.JSONDecodeError:
            return {"1": "Default context"}

    def _generate_image(
        self,
        image_path: str,
        context_option: str,
        context_map: Dict[str, str],
    ) -> Optional[tuple]:
        fallback = lambda: self.generation_tools.generate_entity_image(image_path, context_option)  # noqa: E731
        result = self._run_with_crewai(
            lambda: self._build_generation_crew(context_map),
            {
                "image_path": image_path,
                "entity": self.entity,
                "context_suggestions": json.dumps(context_map),
                "context_option": context_option,
            },
            fallback,
        )
        payload = result.payload or ""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None

        if data.get("status") != "success":
            return None

        base64_image = data.get("image_base64")
        if not base64_image:
            return None

        try:
            raw = base64.b64decode(base64_image)
            pil_image = Image.open(BytesIO(raw))
            pil_image.load()
        except Exception:
            return None

        judge_result = self._judge_image(base64_image)
        return judge_result, pil_image

    def _judge_image(self, image_base64: str) -> bool:
        fallback = lambda: self.judge_tools.judge_generated_image(image_base64)  # noqa: E731
        result = self._run_with_crewai(
            lambda: self._build_judge_crew(image_base64),
            {
                "entity": self.entity,
                "generated_image_base64": image_base64,
            },
            fallback,
        )
        try:
            data = json.loads(result.payload or "{}")
        except json.JSONDecodeError:
            return False
        return bool(data.get("status"))

    def _persist_outputs(self, img_file: str, idx: str, pil_image: Image.Image, report: Dict[str, object]) -> None:
        base_name, ext = os.path.splitext(img_file)
        entity_folder = self.entity_output_folder

        output_filename = f"{base_name}_ctx{idx}{ext}"
        output_path = os.path.join(entity_folder, output_filename)
        pil_image.save(output_path)

        aug_image = augment_image(pil_image)
        aug_filename = f"{base_name}_ctx{idx}_aug{ext}"
        aug_path = os.path.join(entity_folder, aug_filename)
        aug_image.save(aug_path)
        report["augmented_images"] += 1

    # ------------------------------------------------------------------
    def _save_discard(self, img_file: str, idx: str, pil_image: Image.Image) -> None:
        if pil_image is None:
            return
        base_name, _ = os.path.splitext(img_file)
        discard_name = f"{base_name}_ctx{idx}.png"
        discard_path = os.path.join(self.discard_folder, discard_name)
        os.makedirs(self.discard_folder, exist_ok=True)
        pil_image.save(discard_path)

    # ------------------------------------------------------------------
    def clean_discard_folder(self) -> None:
        if os.path.exists(self.discard_folder):
            shutil.rmtree(self.discard_folder)
        os.makedirs(self.discard_folder, exist_ok=True)
