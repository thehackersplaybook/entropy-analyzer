# Entropy Analyzer ⚖️

> 💡 Project is under development and not fit for production use or reliable enough for research yet.

Perform a complete, and detailed entropy analysis on your data. Supports both, real-time and non real-time use-cases.

This is a developer-first product aimed at enhancing human life by helping us understand entropy, it's effects and it's impact on our daily life in a systematic, clear, and AI-driven manner. This package leverages Agentic AI to deliver a detailed, and comprehensive entropy analysis for any kind of data source.

---

## Usage

```bash
pip install entropy-analyzer
```

### CLI Usage

First create a .env file with your `OPENAI_API_KEY`.

```text
OPENAI_API_KEY=test-key
```

Call the CLI with the input and newly created env file.

```bash
python -m entropy_analyzer --input "The mind when focused, can reach the sun" --env .env
```

You can also pass files directly.

```bash
python -m entropy_analyzer -f meeting_audio.mp3 --env .env
python -m entropy_analyzer -f knowledge_base.pdf --env .env
```

### Package Usage

First install the package.

```bash
pip install entropy-analyzer
```

Then, simply import the package and call the `compute` method.

```python
from entropy_analyzer import EntropyAnalyzer

text = "The mind when focused, can reach the sun."

analyzer = EntropyAnalyzer(openai_api_key="test-key", temperature=0.5)

report = analyzer.compute(input=text, structured_output=True)

print(report)
```

You can also stream the response.

```python
from entropy_analyzer import EntropyAnalyzer

text = "The mind when focused, can reach the sun."

analyzer = EntropyAnalyzer(openai_api_key="test-key", temperature=0.5)

response = analyzer.compute(input=text, stream=True)

report_md = ""
report = {}
for chunk in response:
    report_md += chunk.content
    if chunk.is_final:
        report = chunk.report
        break

print({
    "label": f"Entropy Analysis for '{text}'"
    "full_report": report_md,
    "report": report
})
```

---

## Motivation and Objectives

1. Create a research experiment for analyzing entropy across various kinds of data, both from synthetic and real-world sources.
2. Build a re-usable library for entropy analysis that can be used across research studies, experiments, software systems, and prototypes where user value can be enhanced using entropy analysis.
3. Study the relation between entropy and time and the effects of entropy on time in a sandboxed environment.
4. Enhance AI-powered workflows with improved decision-making capabilities and intelligence using real-time entropy analysis of data or knowledge bases.
5. Develop insight into Knowledge Engineering workflows to develop and enhance knowledge systems across The Hackers Playbook, internally, for end-users and clients.

---

## Contributions

We welcome contributions from developers around the globe. The steps to contribute are simple:

- Fork the repository.
- Create a new branch with your changes.
- Submit a PR to this repository.
- Complete the PR review process with our team.

---

## License

Entropy Analyzer is distributed under the MIT License. Refer to the [LICENSE](https://github.com/thehackersplaybook/entropy-analyzer/blob/main/LICENSE) file for full details.
