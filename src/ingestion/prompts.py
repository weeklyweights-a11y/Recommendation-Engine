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

Return ONE JSON object. Include ALL sections below — do not stop after skills.

Required field order in your JSON (most important first):
1. name, email, phone, location (strings or null)
2. experience: array of EVERY job on the resume — at least one entry required
   Each entry: company, title, start_date (YYYY-MM), end_date (YYYY-MM or "present"), duration_months,
   description (max 250 characters), domain, company_size_estimate, company_stage_estimate, role_type, key_achievements
   - company_size_estimate: 1-10, 11-50, 51-200, 201-1000, 1000+
   - company_stage_estimate: pre-seed, seed, series-a, series-b, growth, enterprise
   - role_type: ic, tech-lead, manager, founding, co-founder, executive
3. education: array of degrees (institution, degree, field, graduation_year) — use [] if none listed
4. total_years_experience: number (sum of professional work years; estimate from job dates if not stated)
5. summary: 2-3 sentence career summary (required, non-empty string)
6. domains: array of industry strings
7. role_archetype: founding_builder, platform_engineer, research_scientist, product_engineer, specialist, generalist, manager
8. career_trajectory: ascending, lateral, career_changer, early_career, seasoned
9. inferred_preferences: {{preferred_company_stage, preferred_team_size, preferred_work_style, likely_looking_for}}
10. skills: array — cap at 25 skills; keep each context under 80 characters
   - category: programming_language, framework, tool, platform, methodology, domain_knowledge, soft_skill, other
   - proficiency: beginner, intermediate, advanced, expert
"""

GITHUB_SECTION_TEMPLATE = """
Additionally, here is their GitHub profile data:
{github_summary}
Use this to validate and enhance skill proficiency ratings — if a skill appears in multiple repos with recent commits, rate it higher. Note any skills visible in GitHub but missing from the resume.
"""

RETRY_USER_PROMPT = (
    "Your previous response was incomplete or invalid. Return ONLY valid JSON with ALL required "
    "sections: experience (every job), education, total_years_experience, summary, then skills "
    "(max 25, short context). Do not omit experience or summary."
)

COMPACT_RETRY_PROMPT = (
    "Return compact valid JSON only. Required: experience array (all jobs), education, "
    "total_years_experience, summary, domains, role_archetype, career_trajectory, "
    "inferred_preferences, then up to 20 skills with brief context."
)
