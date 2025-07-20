# Create Issue Command

Create a GitHub issue from Claude Code using the appropriate template.

## Usage
```
/create-issue [issue_type] [title]
```

## Arguments
- `issue_type`: Type of issue to create (required)
- `title`: Title for the issue (required)

## Available Issue Types
- `bug` - Bug report
- `feature` - Feature request
- `performance` - Performance improvement
- `ocr` - OCR accuracy issue
- `reference` - Reference removal issue

## Examples
```
/create-issue bug "OCR fails on mathematical formulas"
/create-issue feature "Add support for new OCR model"
/create-issue performance "Slow processing on large papers"
```

---

Creating GitHub issue: $ARGUMENTS

```bash
# Parse arguments
ARGS=($ARGUMENTS)
ISSUE_TYPE="${ARGS[0]}"
TITLE="${ARGS[@]:1}"

# Validate arguments
if [ -z "$ISSUE_TYPE" ] || [ -z "$TITLE" ]; then
    echo "Usage: /create-issue [issue_type] [title]"
    echo "Available issue types: bug, feature, performance, ocr, reference"
    exit 1
fi

# Map issue type to template
case $ISSUE_TYPE in
    "bug")
        TEMPLATE="bug_report"
        LABELS="bug"
        ;;
    "feature")
        TEMPLATE="feature_request"
        LABELS="enhancement"
        ;;
    "performance")
        TEMPLATE="performance_improvement"
        LABELS="performance"
        ;;
    "ocr")
        TEMPLATE="ocr_accuracy_issue"
        LABELS="ocr,accuracy"
        ;;
    "reference")
        TEMPLATE="reference_removal_issue"
        LABELS="reference-removal,accuracy"
        ;;
    *)
        echo "Unknown issue type: $ISSUE_TYPE"
        echo "Available issue types: bug, feature, performance, ocr, reference"
        exit 1
        ;;
esac

# Create the issue
echo "Creating $ISSUE_TYPE issue: $TITLE"
echo "Template: $TEMPLATE"
echo "Labels: $LABELS"

# Use GitHub CLI to create the issue
gh issue create \
    --title "$TITLE" \
    --template "$TEMPLATE" \
    --label "$LABELS" \
    --web

echo "Issue created successfully!"
echo "You can now fill in the details in the web interface."
```