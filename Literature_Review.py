import os
import pandas as pd
from dotenv import load_dotenv
import anthropic
import time

# --------------- CONFIGURATION ---------------
CSV_INPUT = "scopus_Construction_Cost Estimation.csv"
CSV_CHECKPOINT = "scopus_analysis_results_checkpoint.csv"
N_TEST = 5  # Number of papers for test mode
SLEEP_TIME = 1  # Seconds between API calls
TEST_MODE = False  # Set to False to run the full dataset!
# ---------------------------------------------

# Load API Key
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# Load CSV
df = pd.read_csv(CSV_INPUT)

# If test mode, only process first N_TEST papers
if TEST_MODE:
    df = df.iloc[:N_TEST].copy()

# Load checkpoint if exists, to resume
if os.path.exists(CSV_CHECKPOINT):
    checkpoint_df = pd.read_csv(CSV_CHECKPOINT)
    already_done_ids = set(checkpoint_df['PaperID'])
    print(f"Resuming: {len(already_done_ids)} papers already processed, will skip these.")
    results = checkpoint_df.to_dict(orient="records")
else:
    already_done_ids = set()
    results = []

# ----- Four-category prompt for CSV -----
def make_user_prompt(paper_id, title, abstract):
    return f"""
You are analyzing academic papers about construction cost estimation.

For each paper, analyze the title and abstract and provide for each of the following five analytical categories:
- A binary label ("Yes" or "No") — does the paper belong in this category?
- A short justification (1-2 sentences) explaining your reasoning, referencing details from the title or abstract if possible.
- If the information is not present or cannot be determined confidently, answer "No || Not mentioned or unclear."

The five categories are:
1. Model-Focused Only
Definition: The paper proposes or tests an AI/ML model for cost prediction, optimization, or estimation. It emphasizes model accuracy, algorithmic improvements, or simulation results. Include: ANN, SVM, CNN, regression, fuzzy logic, stacking, etc., but NOT how tools are used in practice.

2. Tool/Platform-Focused
Definition: The paper describes a software system, automation tool, BIM-based estimator, or user-facing platform for construction cost estimation. Include: Revit plugins, quantity takeoff tools, dashboard UI, BIM-5D platforms, AI-assisted interfaces.

3. Real-World Case Study or Application
Definition: The paper includes implementation of a model or tool in an actual construction project, real company setting, field test, or comparison with actual cost data. Include: Any study tested with professionals, site data, pilot studies, industry collaborations.

4. Mentions Human/User Factors
Definition: The paper discusses how people interact with cost estimation tools or methods (training, onboarding, trust, usability, workflow integration, learning curve, estimator perception, etc.). Include: Surveys, interviews, education/training, or explicit references to use/adoption by professionals or students.

5. Primary Focus on Cost Estimation
Does the paper concentrate mainly on construction cost estimation, or is cost estimation a minor or side topic?
Answer: "Yes", "No"

Return the results as a CSV table with exactly six columns:
PaperID,Model-Focused Only,Tool/Platform-Focused,Real-World Case Study or Application,Mentions Human/User Factors,Primary Focus on Cost Estimation

Do not include the title or abstract in the output. Do not include explanations, bullet points, or extra lines. Each cell should contain the Yes/No and justification (e.g., "Yes || Proposes a CNN model..."; "No || Not mentioned or unclear.").

Example output:
PaperID,Model-Focused Only,Tool/Platform-Focused,Real-World Case Study or Application,Mentions Human/User Factors,Primary Focus on Cost Estimation
123,Yes || Proposes a Random Forest model for cost prediction with simulation results.,No || Not mentioned or unclear.,No || Not mentioned or unclear.,No || Not mentioned or unclear.,Yes.

Paper information:
PaperID: {paper_id}
Title: {title}
Abstract: {abstract}
"""


# ------------- Main Loop -------------
for idx, row in df.iterrows():
    paper_id = row['PaperID']
    title = str(row['Title']) if pd.notna(row['Title']) else ""
    abstract = str(row['Abstract']) if pd.notna(row['Abstract']) else ""

    # Skip already processed
    if paper_id in already_done_ids:
        continue

    # Handle missing data
    if not title or not abstract:
        out = {
            "PaperID": paper_id,
            "Model-Focused Only": "No || Not mentioned or unclear.",
            "Tool/Platform-Focused": "No || Not mentioned or unclear.",
            "Real-World Case Study or Application": "No || Not mentioned or unclear.",
            "Mentions Human/User Factors": "No || Not mentioned or unclear."
        }
        results.append(out)
        continue

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.3,
            system="You are an AI assistant that classifies academic abstracts into four analytical categories for a literature review.",
            messages=[
                {"role": "user", "content": make_user_prompt(paper_id, title, abstract)}
            ]
        )
        # Parse Claude's CSV output (ignore header)
        content = response.content[0].text.strip()
        lines = [line for line in content.splitlines() if line.strip()]
        # Find the first row after the header line
        for line in lines:
            if line.startswith(str(paper_id) + ","):
                fields = line.split(",", 5)
                break
        else:  # fallback if not found
            fields = [paper_id] + ["No – Not mentioned or unclear."] * 5

        out = {
            "PaperID": fields[0],
            "Model-Focused Only": fields[1],
            "Tool/Platform-Focused": fields[2],
            "Real-World Case Study or Application": fields[3],
            "Mentions Human/User Factors": fields[4],
            "Primary Focus on Cost Estimation": fields[5]
        }
        results.append(out)

        # Save checkpoint after each paper
        pd.DataFrame(results).to_csv(CSV_CHECKPOINT, index=False)

        print(f"Processed PaperID: {paper_id}")

    except Exception as e:
        print(f"Error on PaperID '{paper_id}': {e}")

    time.sleep(SLEEP_TIME)

# ----- Save Final Output -----
output_df = pd.DataFrame(results, columns=[
    "PaperID",
    "Model-Focused Only",
    "Tool/Platform-Focused",
    "Real-World Case Study or Application",
    "Mentions Human/User Factors",
    "Primary Focus on Cost Estimation"
])

output_df.to_csv(CSV_CHECKPOINT, index=False)
print(f"All done! Results saved in {CSV_CHECKPOINT}")
