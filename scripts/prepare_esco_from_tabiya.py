"""Convert Tabiya ESCO CSVs to standard ESCO filenames for load_esco_neo4j.py."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_TABIYA = (
    Path(__file__).resolve().parent.parent
    / "resources/repos/tabiya-open-dataset/tabiya-esco-v1.1.1/csv"
)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prepare ESCO CSVs from Tabiya format")
    parser.add_argument("--tabiya-dir", type=Path, default=DEFAULT_TABIYA)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    out_dir = Path(settings.paths.esco_data_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tabiya = args.tabiya_dir

    skills = pd.read_csv(tabiya / "skills.csv")
    occupations = pd.read_csv(tabiya / "occupations.csv")
    skill_rels = pd.read_csv(tabiya / "skill_skill_relations.csv")
    occ_skill = pd.read_csv(tabiya / "occupation_skill_relations.csv")

    skill_uri_by_id = dict(zip(skills["ID"], skills["ORIGINURI"]))
    occ_uri_by_id = dict(zip(occupations["ID"], occupations["ORIGINURI"]))

    skills_out = pd.DataFrame(
        {
            "conceptUri": skills["ORIGINURI"],
            "preferredLabel": skills["PREFERREDLABEL"],
            "altLabels": skills["ALTLABELS"],
            "description": skills["DESCRIPTION"],
            "skillType": skills["SKILLTYPE"],
        },
    )
    skills_out.to_csv(out_dir / "skills_en.csv", index=False)
    logger.info("Wrote %s skills", len(skills_out))

    occ_out = pd.DataFrame(
        {
            "conceptUri": occupations["ORIGINURI"],
            "preferredLabel": occupations["PREFERREDLABEL"],
            "description": occupations["DESCRIPTION"],
            "iscoGroup": occupations["ISCOGROUPCODE"],
        },
    )
    occ_out.to_csv(out_dir / "occupations_en.csv", index=False)
    logger.info("Wrote %s occupations", len(occ_out))

    rel_out = pd.DataFrame(
        {
            "originalSkillUri": skill_rels["REQUIRINGID"].map(skill_uri_by_id),
            "relatedSkillUri": skill_rels["REQUIREDID"].map(skill_uri_by_id),
            "relationType": skill_rels["RELATIONTYPE"],
        },
    )
    rel_out = rel_out.dropna(subset=["originalSkillUri", "relatedSkillUri"])
    rel_out.to_csv(out_dir / "skillSkillRelations_en.csv", index=False)
    logger.info("Wrote %s skill relations", len(rel_out))

    occ_skill_out = pd.DataFrame(
        {
            "occupationUri": occ_skill["OCCUPATIONID"].map(occ_uri_by_id),
            "skillUri": occ_skill["SKILLID"].map(skill_uri_by_id),
            "essentialSkill": occ_skill["RELATIONTYPE"].eq("essential"),
        },
    )
    occ_skill_out = occ_skill_out.dropna(subset=["occupationUri", "skillUri"])
    occ_skill_out.to_csv(out_dir / "occupationSkillRelations_en.csv", index=False)
    logger.info("Wrote %s occupation-skill links", len(occ_skill_out))


if __name__ == "__main__":
    main()
