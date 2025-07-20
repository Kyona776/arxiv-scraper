---
name: OCR Accuracy Issue
about: Report OCR processing accuracy problems
title: '[OCR] '
labels: ['ocr', 'accuracy']
assignees: ''

---

## OCR Accuracy Issue Summary
A clear and concise description of the OCR accuracy problem.

## Current Accuracy
- Overall OCR accuracy: [e.g. 85%]
- Specific problem areas: [e.g. mathematical formulas, tables, figures]
- Accuracy requirement: >95%

## OCR Configuration
```yaml
# Paste OCR configuration from config.yaml
ocr:
  model: "mistral_ocr"  # or nougat, unstructured, surya
  device: "cuda"        # or cpu
  # other relevant settings
```

## Paper Information
- ArXiv ID: [e.g. 2301.12345]
- Paper Title: [if known]
- Paper URL: [if using direct PDF]
- File Size: [e.g. 2MB]
- Page Count: [e.g. 15 pages]
- Content Type: [e.g. equations-heavy, table-heavy, figure-heavy]

## Problematic Content
Describe the specific content that's causing accuracy issues:
- [ ] Mathematical equations
- [ ] Chemical formulas
- [ ] Tables
- [ ] Figures with text
- [ ] Captions
- [ ] Headers/footers
- [ ] Multi-column layout
- [ ] Foreign language text
- [ ] Handwritten elements
- [ ] Low-quality scans

## Expected vs. Actual Output
### Expected Output
```
Paste the expected text extraction here
```

### Actual OCR Output
```
Paste the actual OCR output here
```

## Error Analysis
What types of errors are occurring?
- [ ] Character recognition errors
- [ ] Word boundary issues
- [ ] Line break problems
- [ ] Layout detection issues
- [ ] Formula recognition failures
- [ ] Table structure problems
- [ ] Figure caption misplacement
- [ ] Missing text sections
- [ ] Duplicate text sections
- [ ] Incorrect text ordering

## OCR Model Comparison
Have you tested with different OCR models?
- [ ] Mistral OCR: [accuracy/issues]
- [ ] Nougat: [accuracy/issues]
- [ ] Unstructured: [accuracy/issues]
- [ ] Surya: [accuracy/issues]

## Environment Details
- OS: [e.g. macOS, Ubuntu, Windows]
- Hardware: [e.g. M1 Mac, Intel i7, GPU type]
- Python Version: [e.g. 3.9, 3.10]
- CUDA Version: [if using GPU]
- Available Memory: [e.g. 16GB]

## PDF Quality Assessment
- [ ] High quality (crisp text, clear images)
- [ ] Medium quality (some blur, acceptable)
- [ ] Low quality (significant blur, artifacts)
- [ ] Scanned document
- [ ] Generated PDF (text-based)

## Preprocessing Steps
What preprocessing was applied to the PDF?
- [ ] No preprocessing
- [ ] Image enhancement
- [ ] Noise reduction
- [ ] Resolution upscaling
- [ ] Format conversion

## Impact Assessment
How does this accuracy issue affect the pipeline?
- [ ] Prevents proper reference removal
- [ ] Affects LLM extraction quality
- [ ] Causes processing failures
- [ ] Reduces overall accuracy below threshold

## Frequency
How often does this issue occur?
- [ ] Always with this type of content
- [ ] Sometimes (X% of papers)
- [ ] Rarely (edge cases)
- [ ] Only with specific papers

## Proposed Solutions
What solutions might help?
- [ ] Switch to different OCR model
- [ ] Adjust OCR model parameters
- [ ] Add preprocessing steps
- [ ] Implement post-processing correction
- [ ] Use ensemble of multiple OCR models
- [ ] Add content-specific handling

## Testing Requirements
How should the fix be tested?
- [ ] Test with problematic paper samples
- [ ] Accuracy benchmarking
- [ ] Performance impact assessment
- [ ] Regression testing with existing papers
- [ ] Cross-validation with different content types

## Success Criteria
- [ ] OCR accuracy >95% for this content type
- [ ] No regression in other accuracy metrics
- [ ] Processing time within acceptable limits
- [ ] Downstream extraction quality maintained

## Priority
- [ ] Critical - Blocking core functionality
- [ ] High - Significantly impacts extraction quality
- [ ] Medium - Affects some papers
- [ ] Low - Edge case optimization

## Sample Files
If possible, provide sample files or sections that demonstrate the issue:
- [ ] Attached sample PDF
- [ ] Specific page numbers: [e.g. pages 3-5]
- [ ] Specific sections: [e.g. equations in section 2.1]

## Additional Context
Add any other context about the OCR accuracy issue here.

## Logs and Errors
```
Paste relevant log messages and error traces here
```

## Workaround
If you found a temporary workaround, please describe it here.