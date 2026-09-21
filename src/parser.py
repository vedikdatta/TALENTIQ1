"""
Document extraction + structured parsing.

Handles: PDF / DOCX / TXT ingestion, and pulling out
name, email, phone, education, years of experience, and a skills list
from free-form resume or job-description text.
"""

import io
import re
from dataclasses import dataclass, field
from datetime import datetime

from .skills_taxonomy import (
    MASTER_SKILLS,
    SYNONYMS,
    SKILL_TO_CATEGORY,
    canonicalize,
)


EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

URL_RE = re.compile(
    r"(?:https?://[^\s<>\[\]{}\"']+|www\.[^\s<>\[\]{}\"']+|"
    r"(?:https?://)?(?:linkedin\.com|github\.com)/[^\s<>\[\]{}\"']+)",
    re.IGNORECASE,
)


# International-aware phone matcher:
# - optional +country code (1-3 digits)
# - optional parenthesized area/STD code
# - then 2-4 groups of digits separated by space/dot/dash
# - 7-12 digits total
#
# Covers:
#   US "(555) 123-4567"
#   Indian "+91 98765 43210"
#   Indian "+91-9876543210"
#   plain unbroken 10-digit numbers
#
# Still requires digit boundaries so it doesn't grab pieces
# of longer numeric strings such as IDs or years.
PHONE_RE = re.compile(
    r"(?<![\d/])"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{3,5}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
    r"(?![\d/])"
)


# Explicit experience statements such as:
#   3 years experience
#   3 years of experience
#   3+ years relevant experience
#   2 yrs experience
YEARS_EXP_RE = re.compile(
    r"(\d{1,2})\+?\s*"
    r"(?:years|yrs)\.?\s*"
    r"(?:of)?\s*"
    r"(?:relevant\s+)?"
    r"experience",
    re.IGNORECASE,
)


# Employment date ranges such as:
#   2020 - 2024
#   2020 to 2024
#   2020 - Present
#   2020 — Current
DATE_RANGE_RE = re.compile(
    r"(19|20)\d{2}\s*"
    r"(?:-|to|–|—)\s*"
    r"(?:(19|20)\d{2}|present|current)",
    re.IGNORECASE,
)


# A bare "2019-2023"-style year range satisfies the phone digit-count
# shape too; filter those out so employment date ranges never become
# phone numbers.
YEAR_RANGE_LOOKALIKE_RE = re.compile(
    r"^(?:19|20)\d{2}[\s.-]*"
    r"(?:(?:19|20)\d{2}|present|current)$",
    re.IGNORECASE,
)


EDUCATION_LEVELS = [
    ("phd", "PhD / Doctorate"),
    ("doctorate", "PhD / Doctorate"),
    ("master", "Master's Degree"),
    ("mba", "Master's Degree"),
    ("m.s.", "Master's Degree"),
    ("bachelor", "Bachelor's Degree"),
    ("b.s.", "Bachelor's Degree"),
    ("b.tech", "Bachelor's Degree"),
    ("associate", "Associate Degree"),
    ("diploma", "Diploma"),
    ("high school", "High School"),
]


STOPWORD_NAME_LINES = (
    "summary",
    "objective",
    "experience",
    "education",
    "skills",
    "profile",
    "resume",
    "curriculum vitae",
    "highlights",
    "accomplishments",
)


@dataclass
class ParsedDocument:
    raw_text: str
    name: str = "Not detected"
    email: str = "Not detected"
    phone: str = "Not detected"
    linkedin_url: str = "Not detected"
    github_url: str = "Not detected"
    portfolio_url: str = "Not detected"
    education: str = "Not detected"
    years_experience: float = 0.0
    skills: list = field(default_factory=list)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    text_chunks = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""

            # Some PDFs store the destination separately from
            # the visible link text.
            for hyperlink in getattr(page, "hyperlinks", []):
                uri = hyperlink.get("uri")

                if uri:
                    t += f"\n{uri}"

            text_chunks.append(t)

    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(file_bytes))

    text = "\n".join(
        p.text
        for p in doc.paragraphs
    )

    # python-docx exposes hyperlink destinations through
    # document relationships, while paragraph.text contains
    # only their display text.
    links = []

    for relationship in doc.part.rels.values():
        if (
            relationship.is_external
            and relationship.target_ref.startswith(
                ("http://", "https://")
            )
        ):
            if relationship.target_ref not in links:
                links.append(relationship.target_ref)

    if links:
        text += "\n" + "\n".join(links)

    return text


def extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()

    try:
        if lower.endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)

        if lower.endswith(".docx"):
            return extract_text_from_docx(file_bytes)

        # Plain text fallback
        return file_bytes.decode(
            "utf-8",
            errors="ignore",
        )

    except Exception:
        try:
            return file_bytes.decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            return ""


def guess_name(text: str, fallback: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Look through the first 20 meaningful lines.
    for i, line in enumerate(lines[:20]):
        clean = line.strip()

        if not clean or len(clean) > 45:
            continue

        low = clean.lower()

        # Ignore obvious section headings.
        if any(k == low for k in STOPWORD_NAME_LINES):
            continue

        # Ignore contact information.
        if EMAIL_RE.search(clean) or PHONE_RE.search(clean):
            continue

        words = clean.split()

        # Reject obvious job titles / roles.
        title_words = {
            "engineer",
            "developer",
            "scientist",
            "analyst",
            "designer",
            "manager",
            "intern",
            "student",
            "consultant",
            "architect",
            "administrator",
            "specialist",
            "lead",
            "trainee",
        }

        if any(
            word.lower().strip(".,-") in title_words
            for word in words
        ):
            continue

        # Normal case: name is on one line.
        if (
            1 < len(words) <= 4
            and all(
                w.replace(".", "").isalpha() or w.isupper()
                for w in words
            )
        ):
            return clean.title() if clean.isupper() else clean

        # PDF-layout case:
        # name may be split across two consecutive lines,
        # e.g. "AHELI" followed by "BANERJEE".
        if len(words) == 1 and clean.isalpha():
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()

                if (
                    next_line.isalpha()
                    and len(next_line) <= 30
                    and next_line.lower()
                    not in STOPWORD_NAME_LINES
                    and next_line.lower()
                    not in title_words
                ):
                    return (
                        f"{clean.title()} "
                        f"{next_line.title()}"
                    )

    return fallback


def extract_phone(text: str) -> str:
    """
    First PHONE_RE match that isn't actually a
    "2019-2023"-style date range and has enough digits
    to plausibly be a phone number.
    """

    for m in PHONE_RE.finditer(text):
        candidate = m.group(0)

        digit_count = sum(
            ch.isdigit()
            for ch in candidate
        )

        if digit_count < 7:
            continue

        if YEAR_RANGE_LOOKALIKE_RE.match(
            candidate.strip()
        ):
            continue

        return candidate

    return None


def extract_social_links(text: str) -> dict:
    """
    Extract the main professional links from resume text.
    """

    links = []

    for match in URL_RE.finditer(text):
        link = match.group(0).rstrip(
            ".,;:)]}"
        )

        if not link.lower().startswith(
            ("http://", "https://")
        ):
            link = "https://" + link

        if link not in links:
            links.append(link)

    social = {
        "linkedin_url": "Not detected",
        "github_url": "Not detected",
        "portfolio_url": "Not detected",
    }

    remaining = []

    for link in links:
        lower = link.lower()

        if (
            "linkedin.com" in lower
            and social["linkedin_url"] == "Not detected"
        ):
            social["linkedin_url"] = link

        elif (
            "github.com" in lower
            and social["github_url"] == "Not detected"
        ):
            social["github_url"] = link

        else:
            remaining.append(link)

    if remaining:
        social["portfolio_url"] = remaining[0]

    return social


def extract_social_link_labels(text: str) -> dict:
    """
    Find the resume label associated with each extracted
    social link.
    """

    labels = {
        "linkedin_url": "LinkedIn",
        "github_url": "GitHub",
        "portfolio_url": "Portfolio",
    }

    label_re = re.compile(
        r"(linkedin|github|leetcode|portfolio|personal\s+website|"
        r"website|web|site)\s*(?:and\s+links)?\s*:?\s*$",
        re.IGNORECASE,
    )

    for match in URL_RE.finditer(text):
        link = match.group(0).rstrip(
            ".,;:)]}"
        )

        normalized = (
            link
            if link.lower().startswith(
                ("http://", "https://")
            )
            else "https://" + link
        )

        lower = normalized.lower()

        if "linkedin.com" in lower:
            field = "linkedin_url"

        elif "github.com" in lower:
            field = "github_url"

        else:
            field = "portfolio_url"

        line_start = (
            text.rfind("\n", 0, match.start()) + 1
        )

        prefix = text[
            line_start:match.start()
        ]

        label_match = label_re.search(prefix)

        if "leetcode.com" in lower:
            labels["portfolio_url"] = "LeetCode"

        elif label_match:
            label = (
                label_match
                .group(1)
                .strip()
                .lower()
            )

            labels[field] = {
                "linkedin": "LinkedIn",
                "github": "GitHub",
                "leetcode": "LeetCode",
                "portfolio": "Portfolio",
                "personal website": "Personal Website",
                "website": "Website",
                "web": "Web",
                "site": "Site",
            }.get(
                label,
                label.title(),
            )

    return labels


def extract_years_experience(text: str) -> float:
    """
    Extract professional work experience from a resume.

    Rules:
    1. Explicit phrases such as "3 years of experience"
       are accepted.
    2. Date ranges are considered only inside an
       experience/work-history section.
    3. Education dates, graduation dates, project dates,
       certification dates, etc. are ignored.
    4. Fresher/no-experience resumes return 0.0.
    5. Current/present employment uses the actual current year.
    """

    if not text:
        return 0.0

    # ---------------------------------------------------------------
    # 1. Explicit experience statement
    # ---------------------------------------------------------------
    explicit = YEARS_EXP_RE.search(text)

    if explicit:
        try:
            return float(explicit.group(1))
        except (TypeError, ValueError):
            pass

    # ---------------------------------------------------------------
    # 2. Explicit fresher / no-experience indicators
    # ---------------------------------------------------------------
    low_text = text.lower()

    fresher_patterns = [
        r"\bfresher\b",
        r"\bno\s+(?:professional\s+)?experience\b",
        r"\bno\s+work\s+experience\b",
        r"\bno\s+prior\s+experience\b",
        r"\bentry[\s-]?level\b",
        r"\brecently\s+graduated\b",
        r"\brecent\s+graduate\b",
    ]

    for pattern in fresher_patterns:
        if re.search(pattern, low_text):
            return 0.0

    # ---------------------------------------------------------------
    # 3. Find an Experience / Work Experience section
    # ---------------------------------------------------------------
    lines = text.splitlines()

    experience_headers = {
        "experience",
        "work experience",
        "professional experience",
        "employment experience",
        "work history",
        "employment history",
        "professional history",
        "career history",
        "employment",
    }

    education_headers = {
        "education",
        "academic background",
        "academic qualifications",
        "qualifications",
        "certifications",
        "projects",
        "personal projects",
        "skills",
        "technical skills",
        "achievements",
        "awards",
        "interests",
        "references",
    }

    experience_lines = []
    in_experience = False

    for line in lines:
        clean = line.strip()

        if not clean:
            if in_experience:
                experience_lines.append(line)
            continue

        normalized = re.sub(
            r"[:\-|]+$",
            "",
            clean.lower(),
        ).strip()

        # Start of experience section.
        if normalized in experience_headers:
            in_experience = True
            continue

        # Stop when another major section begins.
        if (
            in_experience
            and normalized in education_headers
        ):
            break

        if in_experience:
            experience_lines.append(line)

    experience_text = "\n".join(
        experience_lines
    )

    # ---------------------------------------------------------------
    # 4. If there is no identifiable experience section,
    #    DO NOT guess from arbitrary dates in the resume.
    # ---------------------------------------------------------------
    if not experience_text.strip():
        return 0.0

    # ---------------------------------------------------------------
    # 5. Calculate experience from employment date ranges
    # ---------------------------------------------------------------
    years_found = []

    for match in DATE_RANGE_RE.finditer(
        experience_text
    ):
        span_text = match.group(0)

        all_years = re.findall(
            r"(?:19|20)\d{2}",
            span_text,
        )

        if not all_years:
            continue

        start_year = int(
            all_years[0]
        )

        if (
            "present" in span_text.lower()
            or "current" in span_text.lower()
        ):
            end_year = datetime.now().year

        elif len(all_years) > 1:
            end_year = int(
                all_years[-1]
            )

        else:
            continue

        if end_year >= start_year:
            years_found.append(
                end_year - start_year
            )

    if years_found:
        return float(
            max(years_found)
        )

    return 0.0


def extract_education(text: str) -> str:
    low = text.lower()

    for key, label in EDUCATION_LEVELS:
        if key in low:
            return label

    return "Not detected"


def extract_skills(
    text: str,
    extra_skills: list = None,
) -> list:
    """
    extra_skills lets callers extend the built-in taxonomy
    at runtime (e.g. org-specific tools like LangGraph or
    vLLM added via Settings) without editing skills_taxonomy.py.
    """

    low = (
        " "
        + re.sub(
            r"[^a-z0-9.+#/\s]",
            " ",
            text.lower(),
        )
        + " "
    )

    found = set()

    for skill in MASTER_SKILLS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, low):
            found.add(skill)

    for alias, canonical in SYNONYMS.items():
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(alias)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, low):
            found.add(canonical)

    for skill in (extra_skills or []):
        skill = skill.strip()

        if not skill:
            continue

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, low):
            found.add(skill)

    return sorted(found)


def parse_document(
    filename: str,
    file_bytes: bytes,
    extra_skills: list = None,
) -> ParsedDocument:
    text = extract_text(
        filename,
        file_bytes,
    )

    fallback_name = (
        re.sub(
            r"\.[a-zA-Z0-9]+$",
            "",
            filename,
        )
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    email_match = EMAIL_RE.search(text)
    phone = extract_phone(text)
    social_links = extract_social_links(text)

    return ParsedDocument(
        raw_text=text,
        name=guess_name(
            text,
            fallback_name,
        ),
        email=(
            email_match.group(0)
            if email_match
            else "Not detected"
        ),
        phone=(
            phone
            if phone
            else "Not detected"
        ),
        **social_links,
        education=extract_education(text),
        years_experience=extract_years_experience(
            text
        ),
        skills=extract_skills(
            text,
            extra_skills=extra_skills,
        ),
    )


def parse_job_description(
    jd_text: str,
    min_years_override=None,
    extra_skills: list = None,
) -> dict:
    skills = extract_skills(
        jd_text,
        extra_skills=extra_skills,
    )

    years_match = YEARS_EXP_RE.search(
        jd_text
    )

    min_years = (
        float(years_match.group(1))
        if years_match
        else (
            min_years_override
            or 0.0
        )
    )

    education = extract_education(
        jd_text
    )

    return {
        "raw_text": jd_text,
        "required_skills": skills,
        "min_years": min_years,
        "required_education": education,
    } 