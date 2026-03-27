"""SkillBank: hierarchical skill library for SkillRL.

Loads skills from JSON files, retrieves relevant skills by task category,
and formats them for injection into the agent's system prompt.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .prompts import SKILL_SECTION_TEMPLATE, SKILL_ENTRY_TEMPLATE


class SkillBank:
    """Hierarchical skill library with general + category-specific skills."""

    def __init__(self, skills_path: str):
        """Load skills from a JSON file.

        Args:
            skills_path: Path to JSON file with format:
                {"general_skills": [...], "category_skills": {"cat_name": [...]}}
                OR flat format: {"skills": [...]}
        """
        with open(skills_path) as f:
            data = json.load(f)

        # Support flat format: {"skills": [...]}
        if "skills" in data and "general_skills" not in data:
            self.general_skills: List[dict] = data["skills"]
            self.category_skills: Dict[str, List[dict]] = {}
        else:
            self.general_skills: List[dict] = data.get("general_skills", [])
            self.category_skills: Dict[str, List[dict]] = data.get("category_skills", {})
        self.skills_path = skills_path
        self._domain = Path(skills_path).stem.replace("_skills", "")

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def total_skills(self) -> int:
        cat_count = sum(len(v) for v in self.category_skills.values())
        return len(self.general_skills) + cat_count

    def _format_skill(self, skill: dict) -> str:
        if "when_to_apply" in skill:
            return SKILL_ENTRY_TEMPLATE.format(
                skill_id=skill["skill_id"],
                title=skill["title"],
                principle=skill["principle"],
                when_to_apply=skill["when_to_apply"],
            )
        else:
            return "[{skill_id}] {title}\n  Principle: {principle}".format(
                skill_id=skill["skill_id"],
                title=skill["title"],
                principle=skill["principle"],
            )

    def _format_skill_list(self, skills: List[dict]) -> str:
        return "\n".join(self._format_skill(s) for s in skills)

    def detect_categories(self, text: str) -> List[str]:
        """Detect relevant skill categories from conversation text using keyword matching."""
        text_lower = text.lower()
        matched = []
        for cat_name in self.category_skills:
            # Match category name or close variants
            keywords = [cat_name.lower()]
            # Add common synonyms
            if cat_name.lower() == "cancellation":
                keywords.extend(["cancel", "refund"])
            elif cat_name.lower() == "modification":
                keywords.extend(["modify", "change", "update"])
            elif cat_name.lower() == "booking":
                keywords.extend(["book", "reserve", "new reservation"])
            elif cat_name.lower() == "compensation":
                keywords.extend(["compensat", "certificate", "delayed", "cancelled flight"])
            elif cat_name.lower() == "baggage":
                keywords.extend(["baggage", "luggage", "suitcase", "checked bag"])
            elif cat_name.lower() == "returns":
                keywords.extend(["return", "send back"])
            elif cat_name.lower() == "exchanges":
                keywords.extend(["exchange", "swap", "replace"])
            elif cat_name.lower() == "authentication":
                keywords.extend(["authenticate", "verify", "identity", "who are you"])
            elif cat_name.lower() == "order_modification":
                keywords.extend(["modify", "change item", "update order", "pending"])
            elif cat_name.lower() == "order_status":
                keywords.extend(["status", "where is my", "track"])

            if any(kw in text_lower for kw in keywords):
                matched.append(cat_name)

        return matched

    def get_skills_for_prompt(
        self,
        conversation_text: str = "",
        top_k: int = 6,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Format skills for injection into the system prompt.

        Args:
            conversation_text: Current conversation for category detection.
            top_k: Max total number of skills to include (general + category).
                   For flat-format skill banks, this limits the total directly.
            categories: Override detected categories.

        Returns:
            Formatted skills section string.
        """
        # For flat-format skill banks (no categories), just take top_k from general
        if not self.category_skills:
            skills = self.general_skills[:top_k]
            skills_text = self._format_skill_list(skills)
            return SKILL_SECTION_TEMPLATE.format(
                general_skills=skills_text,
                category_skills="(None)",
            )

        # Hierarchical format: general skills always included
        general_text = self._format_skill_list(self.general_skills)

        # Detect or use provided categories
        if categories is None:
            categories = self.detect_categories(conversation_text)

        # Collect category skills
        cat_skills = []
        for cat in categories:
            if cat in self.category_skills:
                cat_skills.extend(self.category_skills[cat])

        # If no categories matched, include all category skills (sorted by ID)
        if not cat_skills:
            for skills_list in self.category_skills.values():
                cat_skills.extend(skills_list)

        # Top-K limit on category skills
        remaining = max(0, top_k - len(self.general_skills))
        cat_skills = cat_skills[:remaining]
        cat_text = self._format_skill_list(cat_skills) if cat_skills else "(No specific skills matched)"

        return SKILL_SECTION_TEMPLATE.format(
            general_skills=general_text,
            category_skills=cat_text,
        )

    def add_skills(self, new_skills: List[dict]) -> int:
        """Add new skills from evolution step. Returns count added."""
        added = 0
        for skill in new_skills:
            category = skill.get("category", "general")
            if category == "general":
                self.general_skills.append(skill)
            else:
                if category not in self.category_skills:
                    self.category_skills[category] = []
                self.category_skills[category].append(skill)
            added += 1
        return added

    def save(self, path: Optional[str] = None):
        """Save current skills to JSON file."""
        path = path or self.skills_path
        data = {
            "general_skills": self.general_skills,
            "category_skills": self.category_skills,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __repr__(self):
        cat_count = sum(len(v) for v in self.category_skills.values())
        return (f"SkillBank(domain={self._domain}, "
                f"general={len(self.general_skills)}, "
                f"category={cat_count}, "
                f"categories={list(self.category_skills.keys())})")


def load_skillbank_for_domain(domain: str, data_dir: str = None) -> SkillBank:
    """Load the SkillBank for a given domain (airline/retail)."""
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data")
    path = Path(data_dir) / f"{domain}_skills.json"
    if not path.exists():
        raise FileNotFoundError(f"No skill bank found at {path}")
    return SkillBank(str(path))
