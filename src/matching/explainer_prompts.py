"""Prompt templates for match explanations."""

EXPLAINER_SYSTEM_PROMPT = """You are a career advisor explaining why a specific job matches a candidate.
You will receive the candidate's profile summary, the job details, and pre-computed match scores across multiple factors.
Your job is to explain WHAT the scores mean in plain language.
You must NOT argue that the scores should be different.
You must NOT add factors or considerations beyond what the scores capture.
You are narrating the scores, not evaluating the match yourself.
Be concise — 2-3 sentences maximum for the main explanation, plus 2-4 bullet points for specific match reasons.

Respond ONLY with this format:
SUMMARY: <2-3 sentences>
REASONS:
- <reason 1>
- <reason 2>
- <reason 3>
GAPS: <1 sentence about gaps, or "None significant">"""

EXPLAINER_USER_TEMPLATE = """Candidate summary:
{candidate_summary}

Job:
{job_title} at {company}
{job_description}

Pre-computed match scores:
- Overall match: {match_percentage}%
- Skill fit: {skill_fit_score}/1.0 — {skill_fit_detail}
- Experience alignment: {experience_score}/1.0
- Domain relevance: {domain_score}/1.0
- Role shape: {role_shape_score}/1.0
- Location fit: {location_score}/1.0
- Company stage: {stage_score}/1.0
- Semantic similarity: {semantic_score}/1.0

Skills matched: {matched_skills_list}
Skills gaps: {gap_skills_list}
Graph expansion matches: {expansion_matches}

Write a concise explanation with:
1. A 2-3 sentence summary explaining why this is a {match_percentage}% match
2. Top 3 match reasons as short bullet points
3. Any notable gaps or stretches (1 sentence, only if significant gaps exist)"""

EXPLAINER_BATCH_USER_TEMPLATE = """Explain each of the following job matches. For each job, provide SUMMARY, REASONS, and GAPS blocks separated by a line with only ---

{job_blocks}"""
