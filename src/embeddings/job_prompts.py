"""LLM prompts for job field extraction (Gemini Flash)."""

JOB_EXTRACTION_SYSTEM_PROMPT = """You are a job posting analyst. Extract structured fields from this job description.
Respond ONLY with valid JSON matching this schema:
{
  "required_skills": ["list of technical skills, tools, frameworks, languages required"],
  "preferred_skills": ["nice-to-have skills"],
  "domain": "industry or vertical e.g. healthcare, fintech, saas",
  "role_level": "entry|mid|senior|lead|executive",
  "role_type": "ic|tech-lead|manager|founding|executive",
  "responsibilities_summary": "2-3 sentence summary of what the person will do",
  "company_description": "company size, stage, culture, mission if mentioned",
  "team_info": "team size, reporting, cross-functional context",
  "work_style_signals": "pace, methodology, culture signals e.g. fast-paced, collaborative"
}"""

JOB_EXTRACTION_USER_TEMPLATE = """Job title: {title}
Company: {company}

Job description:
{description}
"""

JOB_EXTRACTION_RETRY_PROMPT = """Your previous response was invalid or incomplete JSON.
Return ONLY valid JSON with all required fields for this job posting.
"""
