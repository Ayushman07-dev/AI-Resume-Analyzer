#!/usr/bin/env python3
"""
AI Resume Analyzer & Skill Gap Detector
=======================================
Subject : Programming with Python (01AI0302)

What it does
------------
1. Reads a resume (.txt, .pdf or .docx).
2. Extracts contact details, education, experience and technical skills
   using regular expressions and a skill knowledge base (NLP-style
   keyword extraction with synonym handling).
3. Scores the resume for ATS-friendliness (0-100) with a breakdown.
4. Matches the skills against 10 career roles (weighted matching) and
   ranks the best-fit roles.
5. Detects the SKILL GAP for a target role (or for a pasted job
   description) and builds a prioritised learning roadmap.
6. Saves the full report as .txt and .json.

Only the Python standard library is required.
(Optional: `pip install pypdf` to read PDF resumes.)

Usage
-----
    python ai_resume_analyzer.py                       # interactive menu
    python ai_resume_analyzer.py resume.txt            # best-fit roles
    python ai_resume_analyzer.py resume.txt --role "Data Scientist"
    python ai_resume_analyzer.py resume.txt --jd job_description.txt
"""

import argparse
import html
import json
import math
import os
import re
import sys
import textwrap
import zipfile
from collections import Counter
from datetime import datetime

# =====================================================================
# 1. KNOWLEDGE BASE
# =====================================================================
# category -> skill -> list of aliases (lower case) that mean the same skill
SKILL_DB = {
    "Programming Languages": {
        "Python": ["python", "python3"],
        "Java": ["java"],
        "C": ["c programming", "c language"],
        "C++": ["c++", "cpp"],
        "JavaScript": ["javascript", "es6", "ecmascript"],
        "TypeScript": ["typescript"],
        "Bash": ["bash", "shell scripting", "shell script"],
        "R": ["r programming", "r language"],
    },
    "Core CS": {
        "Data Structures": ["data structures", "data structure", "dsa"],
        "Algorithms": ["algorithms", "algorithm"],
        "OOP": ["oop", "oops", "object oriented", "object-oriented"],
        "Operating Systems": ["operating systems", "operating system"],
        "Computer Networks": ["computer networks", "networking"],
        "DBMS": ["dbms", "database management"],
        "Software Testing": ["unit testing", "pytest", "junit", "selenium",
                             "test cases"],
        "Agile": ["agile", "scrum"],
    },
    "Data & AI": {
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "Machine Learning": ["machine learning", "ml"],
        "Deep Learning": ["deep learning", "neural networks", "cnn", "rnn"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "NLP": ["nlp", "natural language processing", "nltk", "spacy"],
        "Computer Vision": ["computer vision", "opencv"],
        "Statistics": ["statistics", "statistical analysis", "probability"],
        "Data Analysis": ["data analysis", "data analytics", "eda"],
        "Data Visualization": ["data visualization", "data visualisation",
                               "matplotlib", "seaborn", "tableau", "power bi"],
        "Feature Engineering": ["feature engineering"],
        "Excel": ["excel", "ms excel", "microsoft excel"],
        "Generative AI": ["generative ai", "llm", "llms", "langchain"],
    },
    "Web Development": {
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3"],
        "React": ["react", "reactjs", "react.js"],
        "Node.js": ["node.js", "nodejs", "node js"],
        "Flask": ["flask"],
        "Django": ["django"],
        "REST API": ["rest api", "rest apis", "restful", "api development"],
    },
    "Databases": {
        "SQL": ["sql", "mysql", "postgresql", "sqlite", "pl/sql"],
        "MongoDB": ["mongodb", "nosql"],
        "Redis": ["redis"],
    },
    "Cloud & DevOps": {
        "Git": ["git", "version control"],
        "GitHub": ["github"],
        "Docker": ["docker", "containers"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Linux": ["linux", "ubuntu"],
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "GCP": ["gcp", "google cloud"],
        "CI/CD": ["ci/cd", "jenkins", "github actions", "gitlab ci"],
        "Terraform": ["terraform"],
    },
    "Security": {
        "Cybersecurity": ["cybersecurity", "cyber security",
                          "information security"],
        "Ethical Hacking": ["ethical hacking", "penetration testing",
                            "pentesting"],
        "Cryptography": ["cryptography", "encryption"],
        "Wireshark": ["wireshark"],
    },
    "Soft Skills": {
        "Communication": ["communication", "presentation skills"],
        "Teamwork": ["teamwork", "team player", "collaboration"],
        "Problem Solving": ["problem solving", "problem-solving"],
        "Leadership": ["leadership", "team lead"],
    },
}

# role -> {skill: weight}.  3 = must have, 2 = important, 1 = nice to have
ROLES = {
    "Data Scientist": {
        "Python": 3, "Machine Learning": 3, "Statistics": 3, "Pandas": 3,
        "Scikit-learn": 3, "NumPy": 2, "SQL": 2, "Data Visualization": 2,
        "Deep Learning": 2, "Data Analysis": 2, "Feature Engineering": 1,
        "Git": 1, "Communication": 1,
    },
    "Machine Learning Engineer": {
        "Python": 3, "Machine Learning": 3, "Deep Learning": 3,
        "TensorFlow": 2, "PyTorch": 2, "Scikit-learn": 2, "NumPy": 2,
        "Docker": 2, "Git": 2, "Data Structures": 1, "Algorithms": 1,
        "AWS": 1, "NLP": 1,
    },
    "Python Developer": {
        "Python": 3, "OOP": 3, "Git": 3, "Data Structures": 2, "SQL": 2,
        "REST API": 2, "Flask": 2, "Django": 2, "Software Testing": 2,
        "Linux": 1, "Docker": 1, "Problem Solving": 1,
    },
    "Full Stack Web Developer": {
        "HTML": 3, "CSS": 3, "JavaScript": 3, "React": 3, "Node.js": 2,
        "REST API": 2, "SQL": 2, "MongoDB": 2, "Git": 2, "TypeScript": 1,
        "Docker": 1, "Agile": 1,
    },
    "Data Analyst": {
        "SQL": 3, "Excel": 3, "Data Analysis": 3, "Data Visualization": 3,
        "Python": 2, "Pandas": 2, "Statistics": 2, "Communication": 2,
        "Problem Solving": 1,
    },
    "Backend Developer": {
        "SQL": 3, "REST API": 3, "Git": 3, "Python": 2, "Java": 2,
        "Docker": 2, "Data Structures": 2, "Software Testing": 2,
        "MongoDB": 1, "Redis": 1, "Linux": 1, "AWS": 1,
    },
    "DevOps Engineer": {
        "Linux": 3, "Git": 3, "Docker": 3, "Kubernetes": 3, "CI/CD": 3,
        "AWS": 2, "Terraform": 2, "Bash": 2, "Python": 1,
        "Computer Networks": 1, "Azure": 1,
    },
    "Cybersecurity Analyst": {
        "Cybersecurity": 3, "Computer Networks": 3, "Linux": 3,
        "Ethical Hacking": 3, "Cryptography": 2, "Python": 2,
        "Operating Systems": 2, "Wireshark": 2, "Bash": 1,
        "Problem Solving": 1,
    },
    "Cloud Engineer": {
        "AWS": 3, "Linux": 3, "Computer Networks": 2, "Docker": 2,
        "Kubernetes": 2, "Terraform": 2, "Python": 2, "CI/CD": 2,
        "Azure": 1, "GCP": 1,
    },
    "Software Engineer (Fresher)": {
        "Data Structures": 3, "Algorithms": 3, "OOP": 3, "Python": 2,
        "C++": 2, "SQL": 2, "DBMS": 2, "Git": 2, "Operating Systems": 2,
        "Problem Solving": 2, "Communication": 1, "Agile": 1,
    },
}

# skill -> (what to do, estimated weeks to reach working level)
LEARNING = {
    "Python": ("Finish a Python course; build 3 mini projects (CLI apps).", 4),
    "Machine Learning": ("Andrew Ng's ML course; implement 5 algorithms "
                         "with scikit-learn on Kaggle datasets.", 6),
    "Deep Learning": ("Take a deep-learning course; build an image "
                      "classifier with Keras/PyTorch.", 6),
    "Statistics": ("Study descriptive stats, probability, hypothesis "
                   "testing; apply on a real dataset.", 4),
    "Pandas": ("Kaggle 'Pandas' micro-course; clean a messy CSV dataset.", 2),
    "NumPy": ("Practice array operations; re-implement a few formulas.", 1),
    "Scikit-learn": ("Train, tune and evaluate 3 models on Kaggle data.", 3),
    "SQL": ("Practise joins, GROUP BY and subqueries on LeetCode SQL.", 3),
    "Data Visualization": ("Create dashboards with Matplotlib/Seaborn or "
                           "Power BI.", 2),
    "Data Analysis": ("Complete 2 end-to-end EDA notebooks and publish "
                      "them on GitHub.", 3),
    "Excel": ("Learn pivot tables, VLOOKUP/XLOOKUP and charts.", 2),
    "Git": ("Learn commit, branch, merge, pull request; push all projects "
            "to GitHub.", 1),
    "GitHub": ("Create a profile README and pin your 3 best projects.", 1),
    "Docker": ("Containerise one of your projects with a Dockerfile.", 2),
    "Kubernetes": ("Run a small app on Minikube; learn pods/services.", 4),
    "Linux": ("Practise shell commands, permissions and processes.", 3),
    "AWS": ("Prepare for AWS Cloud Practitioner; deploy a static site "
            "on S3/EC2.", 4),
    "CI/CD": ("Add a GitHub Actions pipeline that runs tests on push.", 2),
    "Terraform": ("Provision a VM using Terraform on a free-tier cloud.", 3),
    "HTML": ("Build a 3-page static website.", 1),
    "CSS": ("Learn Flexbox/Grid; clone a landing page.", 2),
    "JavaScript": ("Learn ES6, DOM and fetch API; build a to-do app.", 4),
    "React": ("Build a small app using components, hooks and state.", 4),
    "Node.js": ("Build a REST API with Express.", 3),
    "REST API": ("Build and document a CRUD API with Flask/FastAPI.", 2),
    "Flask": ("Build a Flask app with templates and a database.", 2),
    "Django": ("Follow the official tutorial; build a blog/portal.", 4),
    "MongoDB": ("Store and query JSON documents from Python (pymongo).", 2),
    "Data Structures": ("Solve 50 LeetCode problems on arrays, stacks, "
                        "trees, graphs.", 6),
    "Algorithms": ("Study sorting, searching, recursion, DP basics.", 6),
    "OOP": ("Learn classes, inheritance, polymorphism; refactor a project "
            "into classes.", 2),
    "DBMS": ("Study normalisation, ER diagrams and transactions.", 3),
    "Operating Systems": ("Study processes, threads, scheduling, memory.", 4),
    "Computer Networks": ("Study the OSI/TCP-IP model, DNS, HTTP; use "
                          "Wireshark.", 4),
    "Software Testing": ("Write unit tests with pytest for your projects.", 2),
    "Cybersecurity": ("Do TryHackMe beginner paths; learn the OWASP Top 10.",
                      6),
    "Ethical Hacking": ("Practise labs on TryHackMe / Hack The Box.", 6),
    "Cryptography": ("Learn symmetric/asymmetric encryption and hashing.", 3),
    "TensorFlow": ("Build a neural network with TensorFlow/Keras.", 3),
    "PyTorch": ("Follow the official PyTorch tutorials; train a CNN.", 3),
    "NLP": ("Build a sentiment-analysis project with NLTK/spaCy.", 4),
    "Communication": ("Write a project blog and give a short demo talk.", 2),
    "Teamwork": ("Join a hackathon or open-source project and work in a "
                 "team.", 3),
    "Leadership": ("Lead a college club event or a project team.", 4),
    "Problem Solving": ("Solve one coding problem daily (LeetCode/"
                        "HackerRank).", 4),
    "Agile": ("Learn Scrum basics; manage a project using a Kanban board.",
              1),
    "Generative AI": ("Build a small LLM app using an API and prompt "
                      "engineering.", 3),
}

ACTION_VERBS = {
    "developed", "built", "designed", "implemented", "created", "led",
    "managed", "analyzed", "analysed", "optimized", "optimised", "deployed",
    "automated", "improved", "reduced", "increased", "engineered",
    "collaborated", "trained", "tested", "integrated", "launched",
    "researched", "delivered", "achieved", "organized", "organised",
    "coordinated", "presented", "maintained", "migrated", "configured",
    "modeled", "modelled", "constructed", "established", "streamlined",
}

# words that may appear in a section heading line
SECTION_WORDS = {
    "summary": {"summary", "objective", "profile"},
    "education": {"education", "educational", "qualification",
                  "qualifications"},
    "skills": {"skills"},
    "experience": {"experience", "internship", "internships"},
    "projects": {"projects", "project"},
    "certifications": {"certifications", "certification", "courses"},
    "achievements": {"achievements", "awards", "honors"},
}
HEADING_FILLER = {"and", "work", "professional", "technical", "key", "core",
                  "academic", "personal", "career", "background", "relevant",
                  "other", "extracurricular", "activities"}

STOPWORDS = set("""a an the and or of to in for on with at by from as is are was
were be been this that these those it its we you your our their i my me he she
they them will can may should must have has had do does did not no yes if then
than so such about into over under also more most other some any all each per
etc using use used work working experience years year strong good ability able
required preferred knowledge skills looking candidate role position""".split())

EDU_PATTERNS = {
    "B.Tech / B.E.": r"\bb\.?\s?tech\b|\bb\.?e\.?\b(?=[\s,(-])|bachelor of (?:engineering|technology)",
    "M.Tech / M.E.": r"\bm\.?\s?tech\b|master of (?:engineering|technology)",
    "BCA / B.Sc": r"\bbca\b|\bb\.?\s?sc\b|bachelor of (?:computer applications|science)",
    "MCA / M.Sc": r"\bmca\b|\bm\.?\s?sc\b|master of (?:computer applications|science)",
    "MBA": r"\bmba\b",
    "Diploma": r"\bdiploma\b",
    "Ph.D": r"\bph\.?\s?d\b",
}


# =====================================================================
# 2. FILE READING
# =====================================================================
def read_resume(path):
    """Return the text of a .txt / .pdf / .docx file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("Reading PDF needs pypdf. "
                               "Run: pip install pypdf")
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)

    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        return html.unescape(re.sub(r"<[^>]+>", "", xml))

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# =====================================================================
# 3. INFORMATION EXTRACTION
# =====================================================================
def _build_patterns():
    """Compile one regex per skill (matches any alias as a whole token)."""
    patterns = {}
    for category, skills in SKILL_DB.items():
        for skill, aliases in skills.items():
            alt = "|".join(re.escape(a) for a in
                           sorted(aliases, key=len, reverse=True))
            patterns[skill] = (category, re.compile(
                r"(?<![\w+#])(?:%s)(?![\w+#])" % alt, re.I))
    return patterns


SKILL_PATTERNS = _build_patterns()
C_IN_LIST = re.compile(r"(?:[:,|/]\s*)C(?=\s*(?:[,|/;]|$))", re.M)


def extract_skills(text):
    """Return {skill: number_of_mentions} for every skill found."""
    found = {}
    for skill, (_cat, pat) in SKILL_PATTERNS.items():
        n = len(pat.findall(text))
        if n:
            found[skill] = n
    # A lone "C" is only trusted inside a comma-separated skill list
    # (e.g. "Languages: Python, C, Java") to avoid false matches.
    if "C" not in found and C_IN_LIST.search(text):
        found["C"] = 1
    return found


def group_by_category(skills):
    grouped = {}
    for skill in skills:
        grouped.setdefault(SKILL_PATTERNS[skill][0], []).append(skill)
    return {c: sorted(v) for c, v in grouped.items()}


def extract_contact(text):
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone = None
    for m in re.finditer(r"\+?\d[\d\s-]{8,15}\d", text):
        digits = re.sub(r"\D", "", m.group())
        if 10 <= len(digits) <= 13:
            phone = m.group().strip()
            break
    linkedin = re.search(r"linkedin\.com/in/[\w-]+", text, re.I)
    github = re.search(r"github\.com/[\w-]+", text, re.I)

    name = ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            if len(line.split()) <= 4 and not re.search(r"[\d@]", line):
                name = line.title()
            break
    return {
        "name": name,
        "email": email.group() if email else "",
        "phone": phone or "",
        "linkedin": linkedin.group() if linkedin else "",
        "github": github.group() if github else "",
    }


def extract_education(text):
    low = text.lower()
    degrees = [d for d, p in EDU_PATTERNS.items() if re.search(p, low)]
    cg = re.search(r"(?:cgpa|gpa)\s*[:\-]?\s*(\d+(?:\.\d+)?)", low)
    return degrees, (float(cg.group(1)) if cg else None)


def extract_experience_years(text):
    """Largest 'N years ... experience' value, else 0."""
    best = 0.0
    pat = r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:\w+\s+){0,3}experience"
    for m in re.finditer(pat, text.lower()):
        best = max(best, float(m.group(1)))
    return best


def count_internships(text):
    """Count lines that mention the word 'Intern' (not the heading)."""
    return len(re.findall(r"\bintern\b", text, re.I))


def detect_sections(text):
    found = set()
    for line in text.splitlines():
        words = re.findall(r"[a-z]+", line.lower())
        if not words or len(words) > 4:
            continue
        known = set().union(*SECTION_WORDS.values()) | HEADING_FILLER
        if all(w in known for w in words):
            for sec, keys in SECTION_WORDS.items():
                if keys & set(words):
                    found.add(sec)
    return found


# =====================================================================
# 4. SCORING
# =====================================================================
def ats_score(text, skills, contact, sections, degrees):
    """Return (score out of 100, breakdown dict, list of tips)."""
    words = re.findall(r"\w+", text)
    tips, parts = [], {}

    # Contact details (15)
    pts = 6 * bool(contact["email"]) + 5 * bool(contact["phone"]) \
        + 4 * bool(contact["linkedin"] or contact["github"])
    parts["Contact details"] = (pts, 15)
    if not contact["email"]:
        tips.append("Add a professional email address.")
    if not contact["phone"]:
        tips.append("Add a phone number.")
    if not (contact["linkedin"] or contact["github"]):
        tips.append("Add your LinkedIn and/or GitHub link.")

    # Sections (25)
    weights = {"education": 5, "skills": 5, "projects": 5, "experience": 4,
               "summary": 3, "certifications": 3}
    pts = sum(w for s, w in weights.items() if s in sections)
    parts["Standard sections"] = (pts, 25)
    for s in weights:
        if s not in sections:
            tips.append(f"Add a clear '{s.title()}' section heading.")

    # Skills (15)
    pts = round(min(len(skills) / 12, 1) * 15)
    parts["Skills coverage"] = (pts, 15)
    if len(skills) < 12:
        tips.append("List more relevant technical skills "
                    f"(found {len(skills)}, aim for 12+).")

    # Action verbs (15)
    verbs = {w for w in re.findall(r"[a-z]+", text.lower())
             if w in ACTION_VERBS}
    pts = round(min(len(verbs) / 6, 1) * 15)
    parts["Action verbs"] = (pts, 15)
    if len(verbs) < 6:
        tips.append("Start bullet points with strong action verbs "
                    "(built, designed, implemented...).")

    # Quantified achievements (10)
    nums = re.findall(
        r"\d+(?:\.\d+)?\s*%|[$\u20b9]\s*\d+|\b\d+\+?\s*"
        r"(?:users|records|students|customers|requests|rows|images|"
        r"models|datasets?|projects|members)", text.lower())
    pts = round(min(len(nums) / 3, 1) * 10)
    parts["Quantified results"] = (pts, 10)
    if len(nums) < 3:
        tips.append("Quantify achievements (e.g. 'improved accuracy by "
                    "12%', 'handled 500+ records').")

    # Length (10)
    n = len(words)
    pts = 10 if 250 <= n <= 900 else 5 if 150 <= n < 250 or 900 < n <= 1200 \
        else 2
    parts["Length"] = (pts, 10)
    if n < 250:
        tips.append(f"Resume is short ({n} words). Aim for 250-900 words.")
    elif n > 900:
        tips.append(f"Resume is long ({n} words). Keep it to 1-2 pages.")

    # Education detected (10)
    pts = 10 if degrees else 0
    parts["Education details"] = (pts, 10)
    if not degrees:
        tips.append("Mention your degree/diploma clearly (e.g. B.Tech CSE).")

    total = sum(p for p, _ in parts.values())
    return total, parts, tips


def match_role(found, role_skills):
    """Weighted matching -> (score %, matched list, missing list)."""
    total = sum(role_skills.values())
    got = sum(w for s, w in role_skills.items() if s in found)
    order = sorted(role_skills, key=lambda s: (-role_skills[s], s))
    matched = [s for s in order if s in found]
    missing = [s for s in order if s not in found]
    return round(100 * got / total, 1), matched, missing


def priority_label(weight):
    return {3: "High", 2: "Medium", 1: "Low"}[weight]


def build_roadmap(missing, role_skills):
    plan = []
    for skill in missing:
        action, weeks = LEARNING.get(
            skill, (f"Complete an introductory course on {skill} and build "
                    f"one mini-project.", 3))
        plan.append({"skill": skill,
                     "priority": priority_label(role_skills[skill]),
                     "weeks": weeks, "action": action})
    return plan


def verdict(score):
    if score >= 75:
        return "STRONG MATCH - ready to apply"
    if score >= 50:
        return "MODERATE MATCH - close the gaps below first"
    return "WEAK MATCH - significant upskilling needed"


# ---------- job description analysis ----------
def skills_from_job_description(jd_text):
    """Turn a job description into {skill: weight} using keyword cues."""
    must = re.compile(r"must|required|mandatory|essential|proficien", re.I)
    nice = re.compile(r"preferred|nice to have|plus|bonus|good to have", re.I)
    weights = {}
    for line in jd_text.splitlines():
        for skill, count in extract_skills(line).items():
            w = 3 if must.search(line) else 1 if nice.search(line) else 2
            weights[skill] = max(weights.get(skill, 0), w)
    mentions = extract_skills(jd_text)
    for skill, c in mentions.items():       # repeated skills matter more
        if c >= 3:
            weights[skill] = 3
    return weights


def tokenize(text):
    toks = re.findall(r"[a-z][a-z+#]{1,}", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def cosine_similarity(text_a, text_b):
    """Bag-of-words cosine similarity (0-100)."""
    a, b = Counter(tokenize(text_a)), Counter(tokenize(text_b))
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return round(100 * dot / (na * nb), 1) if na and nb else 0.0


# =====================================================================
# 5. MAIN ANALYSIS PIPELINE
# =====================================================================
def analyze(text, source="resume", role=None, jd_text=None):
    skills = extract_skills(text)
    contact = extract_contact(text)
    degrees, cgpa = extract_education(text)
    sections = detect_sections(text)
    ats, parts, tips = ats_score(text, skills, contact, sections, degrees)

    ranking = []
    for r, req in ROLES.items():
        s, matched, missing = match_role(skills, req)
        ranking.append({"role": r, "score": s, "matched": matched,
                        "missing": missing})
    ranking.sort(key=lambda d: -d["score"])

    result = {
        "generated": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "source": source,
        "profile": {**contact, "education": degrees, "cgpa": cgpa,
                    "experience_years": extract_experience_years(text),
                    "internships": count_internships(text),
                    "word_count": len(re.findall(r"\w+", text))},
        "sections_found": sorted(sections),
        "skills": skills,
        "skills_by_category": group_by_category(skills),
        "ats": {"score": ats, "breakdown": parts, "tips": tips},
        "ranking": ranking,
        "target": None,
        "jd_similarity": None,
    }

    if role:
        req = ROLES[role]
        s, matched, missing = match_role(skills, req)
        result["target"] = {"name": role, "score": s, "matched": matched,
                            "missing": missing, "verdict": verdict(s),
                            "roadmap": build_roadmap(missing, req)}
    elif jd_text:
        req = skills_from_job_description(jd_text)
        if req:
            s, matched, missing = match_role(skills, req)
            result["target"] = {"name": "Job Description", "score": s,
                                "matched": matched, "missing": missing,
                                "verdict": verdict(s),
                                "roadmap": build_roadmap(missing, req)}
        result["jd_similarity"] = cosine_similarity(text, jd_text)
    return result


# =====================================================================
# 6. REPORT FORMATTING
# =====================================================================
def bar(pct, width=20):
    filled = int(round(pct / 100 * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct:.1f}%"


def wrap(label, text, width=68):
    """Wrap 'label : text' with a hanging indent."""
    return textwrap.fill(label + text, width=width,
                         subsequent_indent=" " * len(label))


def build_report(res):
    L = []
    line = "=" * 68
    add = L.append
    add(line)
    add("   AI RESUME ANALYZER & SKILL GAP DETECTOR - REPORT")
    add(line)
    add(f"File      : {res['source']}")
    add(f"Generated : {res['generated']}")

    p = res["profile"]
    add("\n--- CANDIDATE PROFILE " + "-" * 46)
    add(f"Name       : {p['name'] or 'Not detected'}")
    add(f"Email      : {p['email'] or 'Not found'}")
    add(f"Phone      : {p['phone'] or 'Not found'}")
    add(f"LinkedIn   : {p['linkedin'] or 'Not found'}")
    add(f"GitHub     : {p['github'] or 'Not found'}")
    add(f"Education  : {', '.join(p['education']) or 'Not detected'}"
        + (f"  (CGPA {p['cgpa']})" if p["cgpa"] else ""))
    exp = p["experience_years"]
    if exp:
        exp_txt = f"{exp:g} year(s)"
    elif p["internships"]:
        exp_txt = f"Fresher with {p['internships']} internship(s)"
    else:
        exp_txt = "Fresher / not stated"
    add(f"Experience : {exp_txt}")
    add(f"Word count : {p['word_count']}")
    add(wrap("Sections   : ", ", ".join(s.title() for s in
                                        res["sections_found"]) or "None"))

    add("\n--- SKILLS DETECTED (%d) " % len(res["skills"]) + "-" * 42)
    for cat, items in res["skills_by_category"].items():
        add(wrap(f"{cat:<22}: ", ", ".join(items)))

    ats = res["ats"]
    add("\n--- ATS RESUME SCORE " + "-" * 47)
    add(f"Overall: {bar(ats['score'])}")
    for name, (pts, mx) in ats["breakdown"].items():
        add(f"  {name:<20} {pts:>2}/{mx:<2}")
    if ats["tips"]:
        add("\nImprovement tips:")
        for i, t in enumerate(ats["tips"], 1):
            add(textwrap.fill(f"{i}. {t}", width=68, initial_indent="  ",
                              subsequent_indent="     "))

    add("\n--- BEST-FIT CAREER ROLES " + "-" * 42)
    for i, r in enumerate(res["ranking"][:5], 1):
        add(f"{i}. {r['role']:<30} {bar(r['score'])}")

    t = res["target"]
    if t:
        add("\n" + line)
        add(f"   SKILL GAP ANALYSIS  ->  {t['name'].upper()}")
        add(line)
        add(f"Match score : {bar(t['score'])}")
        add(f"Verdict     : {t['verdict']}")
        add(wrap("Skills you HAVE : ", ", ".join(t["matched"]) or "None"))
        add(wrap("Skills you MISS : ", ", ".join(t["missing"]) or "None"))
        if res["jd_similarity"] is not None:
            add(f"Text similarity with job description: "
                f"{res['jd_similarity']}%")
        if t["roadmap"]:
            add("\nLEARNING ROADMAP (highest priority first)")
            add("-" * 68)
            for i, step in enumerate(t["roadmap"], 1):
                add(f"{i:>2}. {step['skill']}  [{step['priority']} priority, "
                    f"~{step['weeks']} weeks]")
                add(textwrap.fill(step["action"], width=68,
                                  initial_indent="    ",
                                  subsequent_indent="    "))
            high = sum(s["weeks"] for s in t["roadmap"]
                       if s["priority"] == "High")
            total = sum(s["weeks"] for s in t["roadmap"])
            add(f"\nEstimated time: ~{high} weeks for High-priority skills, "
                f"~{total} weeks for all gaps.")
    add("\n" + line)
    return "\n".join(L)


def save_reports(res, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(res["source"]))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(out_dir, f"{base}_{stamp}.txt")
    json_path = os.path.join(out_dir, f"{base}_{stamp}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(build_report(res))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return txt_path, json_path


# =====================================================================
# 7. USER INTERFACE (command line)
# =====================================================================
def choose_role():
    names = list(ROLES)
    print("\nAvailable target roles:")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}. {n}")
    while True:
        c = input("Enter role number: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(names):
            return names[int(c) - 1]
        print("Invalid choice, try again.")


def run_and_show(path, role=None, jd_path=None):
    text = read_resume(path)
    if len(text.split()) < 30:
        raise ValueError("Resume text is too short or could not be read.")
    jd_text = read_resume(jd_path) if jd_path else None
    res = analyze(text, source=path, role=role, jd_text=jd_text)
    print(build_report(res))
    txt, js = save_reports(res)
    print(f"Report saved: {txt}\nJSON saved  : {js}")
    return res


def interactive():
    menu = """
+--------------------------------------------------+
|     AI RESUME ANALYZER & SKILL GAP DETECTOR      |
+--------------------------------------------------+
| 1. Analyze resume (find best-fit roles)          |
| 2. Skill gap for a target role                   |
| 3. Skill gap against a job description           |
| 4. Show all roles and required skills            |
| 5. Exit                                          |
+--------------------------------------------------+"""
    while True:
        print(menu)
        choice = input("Choose an option (1-5): ").strip()
        try:
            if choice == "1":
                run_and_show(input("Resume file path: ").strip())
            elif choice == "2":
                path = input("Resume file path: ").strip()
                run_and_show(path, role=choose_role())
            elif choice == "3":
                path = input("Resume file path: ").strip()
                jd = input("Job description file path: ").strip()
                run_and_show(path, jd_path=jd)
            elif choice == "4":
                for r, req in ROLES.items():
                    print(f"\n{r}:")
                    print("  " + ", ".join(f"{s}({w})" for s, w in
                                           req.items()))
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Please enter a number from 1 to 5.")
        except (FileNotFoundError, ValueError, RuntimeError,
                zipfile.BadZipFile) as err:
            print(f"\n[Error] {err}")


def main():
    ap = argparse.ArgumentParser(
        description="AI Resume Analyzer & Skill Gap Detector")
    ap.add_argument("resume", nargs="?", help="resume file (.txt/.pdf/.docx)")
    ap.add_argument("--role", choices=list(ROLES), help="target role")
    ap.add_argument("--jd", help="job description file (.txt/.pdf/.docx)")
    args = ap.parse_args()

    if not args.resume:
        interactive()
        return
    try:
        run_and_show(args.resume, role=args.role, jd_path=args.jd)
    except (FileNotFoundError, ValueError, RuntimeError,
            zipfile.BadZipFile) as err:
        print(f"[Error] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
