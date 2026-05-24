"""LLM prompt templates for resume extraction."""

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert resume analyst and career profiler. Your job is to extract "
    "structured information from resume text and infer career patterns. You must "
    "respond ONLY with valid JSON matching the specified schema. No markdown, no "
    "preamble, no explanation — just the JSON object."
)

EXTRACTION_USER_TEMPLATE = """Extract structured career information from this resume.

Resume text:
{resume_text}
{github_section}

Return a single JSON object with these fields:
- name, email, phone, location (strings or null)
- skills: array of {{name, category, proficiency, years_used, context}}
  - category: programming_language, framework, tool, platform, methodology, domain_knowledge, soft_skill, other
  - proficiency: beginner, intermediate, advanced, expert
- experience: array of {{company, title, start_date, end_date, duration_months, description, domain, company_size_estimate, company_stage_estimate, role_type, key_achievements}}
  - company_size_estimate: 1-10, 11-50, 51-200, 201-1000, 1000+
  - company_stage_estimate: pre-seed, seed, series-a, series-b, growth, enterprise
  - role_type: ic, tech-lead, manager, founding, co-founder, executive
- education: array of {{institution, degree, field, graduation_year}}
- total_years_experience: number
- domains: array of industry strings
- role_archetype: founding_builder, platform_engineer, research_scientist, product_engineer, specialist, generalist, manager
- career_trajectory: ascending, lateral, career_changer, early_career, seasoned
- inferred_preferences: {{preferred_company_stage, preferred_team_size, preferred_work_style, likely_looking_for}}
- summary: 2-3 sentence career summary
"""

GITHUB_SECTION_TEMPLATE = """
Additionally, here is their GitHub profile data:
{github_summary}
Use this to validate and enhance skill proficiency ratings — if a skill appears in multiple repos with recent commits, rate it higher. Note any skills visible in GitHub but missing from the resume.
"""

RETRY_USER_PROMPT = (
    "Your previous response was not valid JSON. Please respond with ONLY a JSON object, "
    "no other text. Match the schema described earlier."
)
