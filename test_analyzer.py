"""Unit tests for the AI Resume Analyzer. Run:  python -m unittest -v"""
import os
import tempfile
import unittest

import ai_resume_analyzer as ra


class TestSkillExtraction(unittest.TestCase):
    def test_aliases_and_symbols(self):
        s = ra.extract_skills("Worked with sklearn, C++, Node.js and K8s.")
        for skill in ("Scikit-learn", "C++", "Node.js", "Kubernetes"):
            self.assertIn(skill, s)

    def test_java_is_not_javascript(self):
        s = ra.extract_skills("I know Java only.")
        self.assertIn("Java", s)
        self.assertNotIn("JavaScript", s)

    def test_git_is_not_github(self):
        s = ra.extract_skills("Hosted code on GitHub.")
        self.assertIn("GitHub", s)
        self.assertNotIn("Git", s)

    def test_lone_c_only_in_list(self):
        self.assertIn("C", ra.extract_skills("Languages: Python, C, Java"))
        self.assertNotIn("C", ra.extract_skills("I scored grade C in art."))


class TestExtraction(unittest.TestCase):
    def test_contact(self):
        c = ra.extract_contact("Asha Patel\nasha@mail.com +91 99887 76655\n"
                               "github.com/asha-p")
        self.assertEqual(c["name"], "Asha Patel")
        self.assertEqual(c["email"], "asha@mail.com")
        self.assertTrue(c["phone"])
        self.assertEqual(c["github"], "github.com/asha-p")

    def test_experience_years(self):
        self.assertEqual(ra.extract_experience_years(
            "Python developer with 3 years of experience"), 3)
        self.assertEqual(ra.extract_experience_years("Fresher"), 0)

    def test_sections_ignore_project_titles(self):
        found = ra.detect_sections("Education\nEducation Management System\n"
                                   "Technical Skills\nProjects")
        self.assertEqual(found, {"education", "skills", "projects"})ṇ


class TestMatching(unittest.TestCase):
    def test_perfect_and_zero(self):
        req = ra.ROLES["Data Analyst"]
        self.assertEqual(ra.match_role(set(req), req)[0], 100.0)
        self.assertEqual(ra.match_role(set(), req)[0], 0.0)

    def test_missing_sorted_by_weight(self):
        _, _, missing = ra.match_role({"Python"}, ra.ROLES["Data Scientist"])
        w = [ra.ROLES["Data Scientist"][m] for m in missing]
        self.assertEqual(w, sorted(w, reverse=True))

    def test_all_role_skills_exist_in_database(self):
        for role, req in ra.ROLES.items():
            for skill in req:
                self.assertIn(skill, ra.SKILL_PATTERNS, f"{skill} in {role}")

    def test_jd_weights(self):
        w = ra.skills_from_job_description(
            "Must have Python.\nNice to have Docker.\nKnowledge of SQL.")
        self.assertEqual((w["Python"], w["Docker"], w["SQL"]), (3, 1, 2))

    def test_cosine(self):
        self.assertEqual(ra.cosine_similarity("python data", "python data"),
                         100.0)
        self.assertEqual(ra.cosine_similarity("python", "cooking"), 0.0)


class TestPipeline(unittest.TestCase):
    def test_full_analysis(self):
        here = os.path.dirname(os.path.abspath(__file__))
        text = ra.read_resume(os.path.join(here, "sample_resume.txt"))
        res = ra.analyze(text, "sample", role="Data Scientist")
        self.assertGreaterEqual(res["ats"]["score"], 80)
        self.assertEqual(res["target"]["name"], "Data Scientist")
        self.assertIn("Statistics", res["target"]["missing"])
        self.assertIn("SKILL GAP ANALYSIS", ra.build_report(res))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            ra.read_resume("no_such_file.txt")

    def test_save_reports(self):
        res = ra.analyze("Python developer. " * 20, "x.txt")
        with tempfile.TemporaryDirectory() as d:
            t, j = ra.save_reports(res, d)
            self.assertTrue(os.path.getsize(t) and os.path.getsize(j))


if __name__ == "__main__":
    unittest.main()
