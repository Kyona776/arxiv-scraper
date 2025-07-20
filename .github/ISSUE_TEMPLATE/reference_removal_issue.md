---
name: Reference Removal Issue
about: Report problems with reference section removal
title: '[REF-REMOVAL] '
labels: ['reference-removal', 'accuracy']
assignees: ''

---

## Reference Removal Issue Summary
A clear and concise description of the reference removal problem.

## Issue Type
- [ ] References not detected/removed
- [ ] Over-deletion (removed non-reference content)
- [ ] Partial removal (some references remain)
- [ ] In-text citations incorrectly removed
- [ ] Performance issue (>3 seconds processing time)

## Current Accuracy
- Reference removal accuracy: [e.g. 95%]
- Accuracy requirement: >99%
- Processing time: [e.g. 5 seconds]
- Time requirement: <3 seconds

## Paper Information
- ArXiv ID: [e.g. 2301.12345]
- Paper Title: [if known]
- Paper URL: [if using direct PDF]
- File Size: [e.g. 2MB]
- Page Count: [e.g. 15 pages]
- Reference Count: [estimated number of references]

## Reference Format
What reference format is used in the paper?
- [ ] IEEE Format (numbered [1], [2], etc.)
- [ ] APA Format (Author, Year)
- [ ] Nature Format (superscript numbers)
- [ ] ArXiv Format (specific arXiv style)
- [ ] Custom/Mixed format
- [ ] Non-English references (specify language)

## Reference Section Details
### Section Header
What is the exact header of the reference section?
```
Paste the reference section header here (e.g. "References", "Bibliography", "参考文献")
```

### Reference Examples
Provide 2-3 examples of references from the paper:
```
[1] Example reference 1...
[2] Example reference 2...
[3] Example reference 3...
```

## Detection Patterns
Which patterns should have been detected?
- [ ] "References" header
- [ ] "Bibliography" header
- [ ] "参考文献" header
- [ ] Numbered list pattern [1], [2], etc.
- [ ] Author-year pattern (Author, 2023)
- [ ] DOI patterns
- [ ] URL patterns
- [ ] ArXiv ID patterns

## Expected vs. Actual Behavior
### Expected Behavior
- [ ] Complete reference section removal
- [ ] Preserve in-text citations
- [ ] Preserve main content
- [ ] Process within 3 seconds

### Actual Behavior
- [ ] References not detected
- [ ] Partial removal only
- [ ] Over-deletion of main content
- [ ] In-text citations removed
- [ ] Processing time exceeded

## Problematic Content
### What was incorrectly removed?
```
Paste content that was incorrectly removed here
```

### What was incorrectly kept?
```
Paste references that should have been removed here
```

## Configuration
```yaml
# Paste reference removal configuration from config.yaml
text_processing:
  remove_references: true
  reference_patterns:
    - "REFERENCES"
    - "Bibliography"
    - "参考文献"
```

## Log Output
```
Paste relevant log messages from reference removal processing
```

## Text Structure Analysis
Describe the paper's text structure:
- [ ] Clear section boundaries
- [ ] Multi-column layout
- [ ] References at end of paper
- [ ] References in multiple sections
- [ ] Appendices after references
- [ ] Footnotes mixed with references

## Pattern Analysis
### Reference Section Location
- Start page: [e.g. page 12]
- End page: [e.g. page 15]
- Approximate line numbers: [if known]

### Reference Format Analysis
- Numbering style: [e.g. [1], (1), 1., etc.]
- Author format: [e.g. Last, First; First Last]
- Year format: [e.g. (2023), 2023]
- Journal format: [e.g. abbreviated, full name]

## Detection Algorithm Issues
Which part of the detection algorithm failed?
- [ ] Section boundary detection
- [ ] Pattern matching
- [ ] Structure analysis
- [ ] Safe deletion validation
- [ ] Quality assurance checks

## Performance Impact
- Processing time: [e.g. 8 seconds]
- Memory usage: [e.g. 500MB]
- CPU usage: [e.g. 100%]
- Impact on downstream processing: [describe]

## Frequency
How often does this issue occur?
- [ ] Always with this reference format
- [ ] Sometimes (X% of papers)
- [ ] Rarely (edge cases)
- [ ] Only with specific paper types

## Proposed Solutions
What solutions might help?
- [ ] Add new detection patterns
- [ ] Improve section boundary detection
- [ ] Enhance pattern matching algorithm
- [ ] Add format-specific handlers
- [ ] Implement better validation
- [ ] Add manual pattern override

## Testing Requirements
How should the fix be tested?
- [ ] Test with problematic paper samples
- [ ] Accuracy benchmarking
- [ ] Performance measurement
- [ ] Regression testing
- [ ] Cross-validation with different formats

## Success Criteria
- [ ] Reference removal accuracy >99%
- [ ] Processing time <3 seconds
- [ ] No over-deletion of main content
- [ ] Preserve all in-text citations
- [ ] Handle this reference format correctly

## Priority
- [ ] Critical - Blocking core functionality
- [ ] High - Significantly impacts extraction quality
- [ ] Medium - Affects some papers
- [ ] Low - Edge case optimization

## Environment Details
- OS: [e.g. macOS, Ubuntu, Windows]
- Python Version: [e.g. 3.9, 3.10]
- Dependencies: [paste relevant versions]

## Sample Content
### Before Reference Removal
```
Paste relevant text sections before reference removal
```

### After Reference Removal
```
Paste the same sections after reference removal
```

## Additional Context
Add any other context about the reference removal issue here.

## Workaround
If you found a temporary workaround, please describe it here.

## Related Issues
Are there any related issues or similar problems?
- Issue #X: [description]
- Similar patterns in other papers: [describe]