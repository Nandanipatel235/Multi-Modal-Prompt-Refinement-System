import uuid
from typing import List, Dict, Any
from pprint import pprint

from click import prompt

class RefinedPrompt:
    def __init__(self):
        self.prompt = {
            "meta": {
                "prompt_id": str(uuid.uuid4()),
                "source_modalities": [],
                "confidence_score": 0.0
            },
            "intent": {
                "summary": None,
                "domain": None,
                "target_users": None
            },
            "functional_requirements": [],
            "non_functional_requirements": {},
            "technical_constraints": [],
            "expected_outputs": [],
            "inputs_provided": {
                "text": False,
                "images": 0,
                "documents": 0
            },
            "assumptions": [],
            "open_questions": [],
            "rejection_reason": None
        }

    def to_dict(self):
        return self.prompt

class TextParser:
    def parse(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        data = {
            "intent": None,
            "functional": [],
            "constraints": [],
            "outputs": []
        }

        if any(word in text_lower for word in ["app", "system", "website", "platform", "tool"]):
            data["intent"] = {
                "summary": text,
                "domain": "software",
                "target_users": "general"
            }

        if "track" in text_lower:
            data["functional"].append("Track user data")

        if "report" in text_lower:
            data["functional"].append("Generate reports")

        if "mobile" in text_lower:
            data["constraints"].append("Mobile platform")

        if "dashboard" in text_lower:
            data["outputs"].append("Dashboard UI")

        return data

class ImageParser:
    def parse(self, images: List[str]) -> Dict[str, Any]:
        # In real system: Vision model / OCR
        return {
    "intent": {
        "summary": "Visual design reference provided",
        "domain": "design",
        "target_users": None
    },
       "functional": ["Visual layout matching reference"],
       "outputs": ["UI design"]
    }

class DocumentParser:
    def parse(self, documents: List[str]) -> Dict[str, Any]:
        # In real system: PDFMiner / docx / OCR
        return {
            "intent": {
                "summary": "System based on provided documentation",
                "domain": "technical system",
                "target_users": None
            },
            "functional": ["Implement features described in document"],
            "constraints": ["Follow documented specifications"]
        }

class SemanticNormalizer:
    def normalize(self, parsed_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = {
            "intent": None,
            "functional": [],
            "constraints": [],
            "outputs": [],
            "assumptions": [],
            "questions": []
        }

        for item in parsed_inputs:
            if not normalized["intent"] and item.get("intent"):
                normalized["intent"] = item.get("intent")

            normalized["functional"].extend(item.get("functional", []))
            normalized["constraints"].extend(item.get("constraints", []))
            normalized["outputs"].extend(item.get("outputs", []))

        if not normalized["intent"]:
            normalized["questions"].append(
                "What is the main purpose of the requested system?"
            )

        return normalized

class PromptRefiner:
    def refine(self, normalized: Dict[str, Any], input_meta: Dict[str, Any]) -> Dict[str, Any]:
        prompt = RefinedPrompt()

        prompt.prompt["inputs_provided"] = input_meta
        prompt.prompt["meta"]["source_modalities"] = [
            k for k, v in input_meta.items() if v
        ]

        # Rejection logic
        if not normalized["intent"]:
             prompt.prompt["open_questions"].append(
        "Intent is unclear. Please provide a short description of the goal."
    )
        prompt.prompt["meta"]["confidence_score"] = 0.2
        return prompt.to_dict()

        prompt.prompt["intent"] = normalized["intent"]
        prompt.prompt["functional_requirements"] = list(set(normalized["functional"]))
        prompt.prompt["technical_constraints"] = list(set(normalized["constraints"]))
        prompt.prompt["expected_outputs"] = list(set(normalized["outputs"]))
        prompt.prompt["assumptions"] = normalized["assumptions"]
        prompt.prompt["open_questions"] = normalized["questions"]

        # Confidence calculation
        score = 0.4
        if prompt.prompt["functional_requirements"]:
            score += 0.3
        if prompt.prompt["expected_outputs"]:
            score += 0.3

        prompt.prompt["meta"]["confidence_score"] = round(score, 2)

        return prompt.to_dict()

class MultiModalPromptRefinementSystem:
    def __init__(self):
        self.text_parser = TextParser()
        self.image_parser = ImageParser()
        self.document_parser = DocumentParser()
        self.normalizer = SemanticNormalizer()
        self.refiner = PromptRefiner()

    def process(
        self,
        text: str = None,
        images: List[str] = None,
        documents: List[str] = None
    ) -> Dict[str, Any]:

        parsed = []
        meta = {
            "text": False,
            "images": 0,
            "documents": 0
        }

        if text:
            parsed.append(self.text_parser.parse(text))
            meta["text"] = True

        if images:
            parsed.append(self.image_parser.parse(images))
            meta["images"] = len(images)

        if documents:
            parsed.append(self.document_parser.parse(documents))
            meta["documents"] = len(documents)

        normalized = self.normalizer.normalize(parsed)
        return self.refiner.refine(normalized, meta)

if __name__ == "__main__":
    system = MultiModalPromptRefinementSystem()

    print("\n--- Example 1: Text Only ---")
    pprint(system.process(
        text="Build a mobile app to track expenses and generate monthly reports"
    ))

    print("\n--- Example 2: Text + Image ---")
    pprint(system.process(
        text="Design a bakery website",
        images=["bakery_ui.png"]
    ))

    print("\n--- Example 3: Document Only ---")
    pprint(system.process(
        documents=["specification.pdf"]
    ))

    print("\n--- Example 4: Image Only (Incomplete) ---")
    pprint(system.process(
        images=["dashboard.png"]
    ))

    print("\n--- Example 5: Irrelevant Input ---")
    pprint(system.process())