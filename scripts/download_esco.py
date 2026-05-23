"""Download or validate ESCO taxonomy CSV files."""

import argparse
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings

logger = logging.getLogger(__name__)

REQUIRED_PATTERNS = (
    "skills_en.csv",
    "occupations_en.csv",
)


def find_file(data_dir: Path, pattern: str) -> Path | None:
    """Find first file matching pattern recursively."""
    matches = list(data_dir.rglob(pattern))
    return matches[0] if matches else None


def validate_files(data_dir: Path) -> bool:
    """Validate ESCO CSV presence and row counts."""
    ok = True
    skills = find_file(data_dir, "skills_en.csv")
    occupations = find_file(data_dir, "occupations_en.csv")
    if not skills:
        logger.error("Missing skills_en.csv under %s", data_dir)
        ok = False
    else:
        lines = sum(1 for _ in skills.open(encoding="utf-8", errors="ignore")) - 1
        logger.info("skills_en.csv: ~%s rows", lines)
        if lines < 10000:
            logger.warning("Expected at least 10000 skills, found %s", lines)
    if not occupations:
        logger.error("Missing occupations_en.csv under %s", data_dir)
        ok = False
    else:
        lines = sum(1 for _ in occupations.open(encoding="utf-8", errors="ignore")) - 1
        logger.info("occupations_en.csv: ~%s rows", lines)
        if lines < 3000:
            logger.warning("Expected at least 3000 occupations, found %s", lines)
    relations = find_file(data_dir, "skillSkillRelations_en.csv") or find_file(
        data_dir,
        "broaderRelationsSkillPillar_en.csv",
    )
    if not relations:
        logger.warning("No skill relation CSV found (skillSkillRelations or broaderRelations)")
    return ok


def print_manual_instructions(data_dir: Path) -> None:
    """Print manual download instructions."""
    logger.info(
        "Download ESCO CSV (English) from https://esco.ec.europa.eu/en/use-esco/download "
        "and extract into: %s",
        data_dir,
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate or prepare ESCO CSV data")
    parser.add_argument("--force", action="store_true", help="Re-validate after download")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    data_dir = Path(settings.paths.esco_data_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        logger.info("Force flag set — re-validating files")

    if not validate_files(data_dir):
        print_manual_instructions(data_dir)
        sys.exit(1)

    logger.info("ESCO CSV validation passed")


if __name__ == "__main__":
    main()
