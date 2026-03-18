"""
Assignment Name: ML Idea Generator
Description: Suggest ML problems in college, healthcare, shopping and describe input -> output.
"""

from dataclasses import dataclass
from typing import List
import argparse


@dataclass
class MLIdea:
    domain: str
    problem: str
    input_data: str
    output_data: str
    model_hint: str


def get_ml_ideas() -> List[MLIdea]:
    return [
        MLIdea(
            domain="college",
            problem="Student dropout risk prediction",
            input_data="Attendance %, assignment scores, quiz marks, LMS activity, past semester GPA",
            output_data="Risk score (0-1) and class label: low/medium/high dropout risk",
            model_hint="Classification (Logistic Regression, Random Forest, XGBoost)",
        ),
        MLIdea(
            domain="college",
            problem="Course recommendation for next semester",
            input_data="Completed courses, grades, interests, prerequisites, timetable constraints",
            output_data="Ranked list of recommended courses",
            model_hint="Recommendation system (content-based + collaborative filtering)",
        ),
        MLIdea(
            domain="college",
            problem="Campus placement package prediction",
            input_data="CGPA, coding test score, project quality score, communication rating, internship count",
            output_data="Predicted salary range or expected package",
            model_hint="Regression (Linear Regression, Gradient Boosting)",
        ),
        MLIdea(
            domain="healthcare",
            problem="Early diabetes detection",
            input_data="Age, BMI, glucose level, blood pressure, family history, activity level",
            output_data="Probability of diabetes and binary decision (positive/negative)",
            model_hint="Binary classification (SVM, Random Forest, Neural Network)",
        ),
        MLIdea(
            domain="healthcare",
            problem="Hospital readmission prediction",
            input_data="Diagnosis codes, medications, length of stay, prior admissions, lab results",
            output_data="Chance of readmission within 30 days",
            model_hint="Classification with imbalance handling (XGBoost, LightGBM)",
        ),
        MLIdea(
            domain="healthcare",
            problem="Patient triage priority estimation",
            input_data="Symptoms text, vitals (heart rate, SpO2, BP), age, chronic conditions",
            output_data="Urgency level: critical/high/medium/low",
            model_hint="Multiclass classification + NLP (BERT + tabular model)",
        ),
        MLIdea(
            domain="shopping",
            problem="Product recommendation engine",
            input_data="User clicks, purchases, cart history, product metadata, ratings",
            output_data="Top-N personalized product recommendations",
            model_hint="Recommendation (matrix factorization, deep ranking model)",
        ),
        MLIdea(
            domain="shopping",
            problem="Demand forecasting for inventory",
            input_data="Past sales by date, promotions, seasonality, holidays, stock levels",
            output_data="Forecasted item demand for upcoming days/weeks",
            model_hint="Time series forecasting (Prophet, LSTM, XGBoost)",
        ),
        MLIdea(
            domain="shopping",
            problem="Review sentiment classification",
            input_data="Customer review text, rating, product category",
            output_data="Sentiment label: positive/neutral/negative",
            model_hint="Text classification (TF-IDF + Logistic Regression, Transformer)",
        ),
    ]


def print_ideas(ideas: List[MLIdea]) -> None:
    current_domain = ""
    for idx, idea in enumerate(ideas, start=1):
        if idea.domain != current_domain:
            current_domain = idea.domain
            print(f"\n=== {current_domain.upper()} ===")

        print(f"\n{idx}. Problem: {idea.problem}")
        print(f"   Input  -> {idea.input_data}")
        print(f"   Output -> {idea.output_data}")
        print(f"   Model  -> {idea.model_hint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML project ideas by domain.")
    parser.add_argument(
        "--domain",
        default="all",
        choices=["all", "college", "healthcare", "shopping"],
        help="Filter ideas by domain",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ideas = get_ml_ideas()

    if args.domain != "all":
        ideas = [idea for idea in ideas if idea.domain == args.domain]

    if not ideas:
        print("No ideas found for the selected domain.")
        return

    print_ideas(ideas)


if __name__ == "__main__":
    main()
