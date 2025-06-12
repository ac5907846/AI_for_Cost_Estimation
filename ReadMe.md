# API Key Setup
This project uses Anthropic’s Claude API. To keep your credentials secure, the API key is loaded from a .env file (not committed to git).

# Get your API key
Sign up and obtain your API key from Anthropic's dashboard: [link](https://console.anthropic.com/settings/keys)

# Cost of Claude Models

Costs of API model can be found here: [link](https://docs.anthropic.com/en/docs/about-claude/pricing)

# Transcripts

The transcript files used in this analysis contain sensitive interview data from research participants. For access to transcript data, please contact the researchers for more information.

The analysis expects PDF transcript files named A.pdf, B.pdf, C.pdf, etc., to be placed in a transcripts/ folder.

# Create a .env file
In your project root, create a file named .env with the following content:

```
ANTHROPIC_API_KEY=sk-your-anthropic-key-here