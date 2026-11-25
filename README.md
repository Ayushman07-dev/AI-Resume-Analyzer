# AI Resume Analyzer & Skill Gap Detector

A Python command-line application for resume evaluation, career guidance and placement preparation.

## Features

- Read TXT, PDF and DOCX resumes.
- Extract technical skills and candidate information using regular expressions and keyword matching.
- Calculate a heuristic ATS-friendliness score out of 100.
- Rank 10 career roles using weighted skill matching.
- Identify missing skills for a target role or job description.
- Generate a prioritized learning roadmap.
- Export reports in TXT and JSON formats.

## Getting started

```bash
git clone https://github.com/Ayushman07-dev/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
python -m pip install -r requirements.txt
python ai_resume_analyzer.py
```

The core application uses the Python standard library. The optional pypdf dependency enables PDF resume extraction.

## Example commands

```bash
python ai_resume_analyzer.py sample_resume.txt
python ai_resume_analyzer.py sample_resume.txt --role "Data Scientist"
python ai_resume_analyzer.py sample_resume.txt --jd sample_job_description.txt
python -m unittest test_analyzer -v
```

## Sample output

The supplied sample resume produced an ATS score of 95/100, 17 detected skills and a Data Analyst role match of 76.2%.

## Project files

| File | Purpose |
| --- | --- |
| ai_resume_analyzer.py | Main application |
| test_analyzer.py | Unit tests |
| sample_resume.txt | Example resume |
| sample_resume_weak.txt | Example resume for comparison |
| sample_job_description.txt | Example job description |
| [Project report](AI_Resume_Analyzer_Project_Report.pdf) | Project documentation |

## Limitations

This is a rule-based NLP-style educational project, not a trained machine learning model or a commercial ATS. Scores are estimates and keyword matching may miss contextual information. Avoid committing real resumes or generated reports containing personal information.

## Author

Ayushman Kumar — B.Tech CSE (AI & ML)
