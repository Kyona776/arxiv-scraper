---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug']
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Screenshots/Logs
If applicable, add screenshots or error logs to help explain your problem.

## Environment
- OS: [e.g. macOS, Ubuntu, Windows]
- Python Version: [e.g. 3.9, 3.10]
- Dependencies: [paste relevant versions from requirements.txt]
- OCR Model: [e.g. Mistral OCR, Nougat, etc.]

## Paper Information (if applicable)
- ArXiv ID: [e.g. 2301.12345]
- Paper Title: [if known]
- Paper URL: [if using direct PDF]
- File Size: [if using local PDF]

## Configuration
```yaml
# Paste relevant configuration from config.yaml
```

## Error Messages
```
Paste full error messages and stack traces here
```

## Performance Impact
- [ ] This bug affects processing time
- [ ] This bug affects accuracy
- [ ] This bug affects memory usage
- [ ] This bug causes crashes

## Component Affected
- [ ] OCR Processing (`ocr_processor.py`)
- [ ] Reference Removal (`reference_cleaner.py`)
- [ ] LLM Extraction (`llm_extractor.py`)
- [ ] ArXiv API (`arxiv_api.py`)
- [ ] Text Processing (`text_processor.py`)
- [ ] CSV Generation (`csv_generator.py`)
- [ ] Main Pipeline (`arxiv_extractor.py`)

## Reproducibility
- [ ] This bug happens every time
- [ ] This bug happens sometimes
- [ ] This bug happened once
- [ ] I can't reproduce this bug

## Workaround
If you found a temporary workaround, please describe it here.

## Additional Context
Add any other context about the problem here.

## Priority Assessment
- [ ] Critical - Blocks core functionality
- [ ] High - Significantly impacts performance or accuracy
- [ ] Medium - Affects some functionality
- [ ] Low - Minor issue or edge case