import os
import csv
import anthropic
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("No API key found. Please set ANTHROPIC_API_KEY in your .env file or environment.")

# Folder containing transcripts
TRANSCRIPTS_FOLDER = "transcripts"
OUTPUT_CSV = "quotes_and_themes.csv"

# Context of the study
STUDY_CONTEXT = """
This analysis is part of a research study examining how AI is transforming construction cost estimation education and practice. The study explores professional estimators' experiences with AI tools (particularly Togal.AI), their recommendations, desires, and visions for future AI development in cost estimation and construction management.

The interview transcripts are from professional cost estimators with a wide range of experience (from interns to 35+ year veterans), representing different company types—including professional cost estimation consultants and general contractors in commercial and heavy construction.

Please extract exact quotes (in quotation marks) that provide insights about:
- Recommendations, desires, and visions for future AI development in construction cost estimation.
- Current workflow challenges, pain points, or needs—even if not directly related to AI, as these may inform AI development priorities.
- Explicit technical details, feature requests, timeline expectations, or adoption barriers.

When extracting quotes:
- Ensure each quote has enough context for clarity (not too short or too long; include surrounding sentences if needed, especially for pronouns like "it" or "this").
- Select quotes that represent complete thoughts or ideas.
- Identify both explicit AI-related insights and implicit industry needs.

Both AI-specific commentary and general challenges are valuable for this research.
"""

# Claude prompt
PROMPT_TEMPLATE = """
You are analyzing a transcript from an interview with a professional construction cost estimator.
{context}

TASKS:
1. Extract direct quotes from the transcript that are relevant to the topic above.
2. For each quote, assign a concise theme (one sentence or phrase) that summarizes the topic or insight.

Provide your response in this exact format:

THEME: [Theme 1]
QUOTE: "[Direct quote 1]"

THEME: [Theme 2]
QUOTE: "[Direct quote 2]"

(Repeat for all themes/quotes)
"""

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def query_claude(estimator, transcript_text):
    prompt = PROMPT_TEMPLATE.format(context=STUDY_CONTEXT) + "\n\nTranscript:\n" + transcript_text
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text if hasattr(response.content[0], 'text') else response.content[0]


def parse_response(response_text):
    quotes_and_themes = []
    lines = response_text.split('\n')
    theme, quote = None, None
    for line in lines:
        if line.startswith("THEME:"):
            theme = line.replace("THEME:", "").strip()
        elif line.startswith("QUOTE:"):
            quote = line.replace("QUOTE:", "").strip().strip('"')
            if theme and quote:
                quotes_and_themes.append((theme, quote))
                theme, quote = None, None
    return quotes_and_themes

def main():
    all_data = []
    for filename in os.listdir(TRANSCRIPTS_FOLDER):
        if filename.endswith(".pdf"):
            estimator = filename.replace('.pdf', '')
            pdf_path = os.path.join(TRANSCRIPTS_FOLDER, filename)
            print(f"Processing {filename} ...")
            text = extract_text_from_pdf(pdf_path)
            claude_response = query_claude(estimator, text)
            print(f"Claude Response for {estimator}:\n{claude_response}\n")
            parsed = parse_response(claude_response)
            for theme, quote in parsed:
                all_data.append([estimator, theme, quote])
    
    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Estimator', 'Theme', 'Quote'])
        writer.writerows(all_data)
    print(f"Done! Extracted data written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
