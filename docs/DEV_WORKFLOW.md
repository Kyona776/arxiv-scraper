# Development Workflow Guide - ArXiv Scraper Pipeline

## Overview

This guide establishes a comprehensive development workflow for the ArXiv Scraper Pipeline project, integrating GitHub Projects, Issues, Pull Requests, and Claude Code slash commands for efficient development and collaboration.

## Table of Contents

1. [GitHub Project Management](#github-project-management)
2. [Issue Management](#issue-management)
3. [Pull Request Workflow](#pull-request-workflow)
4. [Branch Strategy](#branch-strategy)
5. [Claude Code Integration](#claude-code-integration)
6. [Testing and Quality Assurance](#testing-and-quality-assurance)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Performance Monitoring](#performance-monitoring)
9. [Release Management](#release-management)

## GitHub Project Management

### Project Board Setup

Our GitHub Project uses a Kanban-style board with the following columns:

- **Backlog**: New issues and feature requests
- **Ready**: Issues that are ready to be worked on
- **In Progress**: Currently being developed
- **Review**: Code review and testing
- **Done**: Completed and deployed

### Project Automation Rules

1. **Auto-move to "In Progress"** when PR is opened
2. **Auto-move to "Review"** when PR is ready for review
3. **Auto-move to "Done"** when PR is merged
4. **Auto-assign labels** based on issue type and component

### Project Views

- **Overview**: High-level project status
- **By Component**: Grouped by module (OCR, Reference Removal, LLM, etc.)
- **By Priority**: Critical, High, Medium, Low
- **Sprint View**: Current sprint progress

## Issue Management

### Issue Types and Labels

#### Core Labels
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `performance`: Performance optimization
- `documentation`: Documentation improvements
- `security`: Security-related issues

#### Component Labels
- `ocr`: OCR processing module
- `reference-removal`: Reference cleaning module
- `llm-extraction`: LLM analysis module
- `api`: ArXiv API integration
- `testing`: Testing infrastructure
- `ci-cd`: Continuous integration/deployment

#### Priority Labels
- `priority:critical`: Blocking issues
- `priority:high`: Important features/fixes
- `priority:medium`: Standard development
- `priority:low`: Nice-to-have improvements

### Issue Templates

Use the following issue templates located in `.github/ISSUE_TEMPLATE/`:

1. **Bug Report**: For reporting bugs and issues
2. **Feature Request**: For requesting new features
3. **Performance Issue**: For performance-related problems
4. **OCR Accuracy Issue**: Specific to OCR processing problems
5. **Reference Removal Issue**: Specific to reference cleaning problems

### Issue Workflow

1. **Create Issue**: Use appropriate template
2. **Triage**: Assign labels, priority, and component
3. **Assignment**: Assign to developer or team
4. **Development**: Create branch and implement
5. **Review**: Create PR and request review
6. **Closure**: Merge PR and close issue

## Pull Request Workflow

### PR Requirements

All PRs must include:
- Clear description of changes
- Link to related issue(s)
- Testing evidence
- Performance impact assessment
- Documentation updates (if applicable)

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Security considerations addressed
- [ ] Breaking changes documented

### Review Process

1. **Self-Review**: Author reviews their own code
2. **Automated Checks**: CI/CD pipeline runs
3. **Peer Review**: At least one team member reviews
4. **Performance Review**: For performance-critical changes
5. **Final Approval**: Project maintainer approval

### Merge Requirements

- All CI checks pass
- At least one approved review
- No merge conflicts
- Up-to-date with main branch

## Branch Strategy

### Git Flow

We use a simplified Git Flow with the following branches:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

### Branch Naming Convention

- `feature/issue-123-add-new-ocr-model`
- `bugfix/issue-456-fix-reference-removal`
- `hotfix/critical-memory-leak`
- `release/v1.2.0`

### Branch Protection Rules

- `main`: Requires PR review, status checks, and up-to-date branch
- `develop`: Requires status checks and up-to-date branch

## Claude Code Integration

### Built-in Commands

Use these Claude Code slash commands for common tasks:

- `/review`: Request code review
- `/bug`: Report bugs
- `/status`: Check project status
- `/test`: Run tests
- `/config`: View configuration

### Custom Slash Commands

Located in `.claude/commands/`, these custom commands streamline development:

#### `/test-module [module_name]`
Run tests for specific modules:
```bash
/test-module reference_cleaner
/test-module ocr_processor
```

#### `/check-performance`
Run performance benchmarks:
```bash
/check-performance
```

#### `/create-issue [type] [title]`
Create GitHub issue from Claude:
```bash
/create-issue bug "OCR accuracy degradation"
/create-issue feature "Add new OCR model support"
```

#### `/setup-dev`
Set up development environment:
```bash
/setup-dev
```

#### `/run-pipeline [paper_id]`
Execute full pipeline test:
```bash
/run-pipeline 2301.12345
```

#### `/check-refs [file_path]`
Test reference removal accuracy:
```bash
/check-refs data/sample_paper.pdf
```

### Claude Code Best Practices

1. **Use descriptive commit messages** with issue references
2. **Run tests before committing** using `/test-module`
3. **Check performance impact** with `/check-performance`
4. **Create issues for bugs** found during development
5. **Use `/review` for code reviews** before merging

## Testing and Quality Assurance

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Module interaction testing
3. **Performance Tests**: Benchmarking and profiling
4. **Accuracy Tests**: OCR and extraction accuracy
5. **End-to-End Tests**: Full pipeline testing

### Test Commands

```bash
# Run all tests
uv run pytest tests/

# Run specific module tests
uv run pytest tests/test_reference_cleaner.py

# Run with coverage
uv run pytest --cov=src tests/

# Run performance tests
uv run pytest tests/test_performance.py -v

# Run accuracy tests
uv run pytest tests/test_accuracy.py -v
```

### Quality Gates

All code must pass:
- Unit tests (>90% coverage)
- Integration tests
- Performance benchmarks
- Code style checks (black, flake8)
- Security scans

## CI/CD Pipeline

### GitHub Actions Workflows

1. **Test Workflow**: Runs on every PR
   - Unit and integration tests
   - Code coverage reporting
   - Performance benchmarking

2. **Quality Workflow**: Code quality checks
   - Code style validation
   - Security scanning
   - Dependency vulnerability checks

3. **Release Workflow**: Automated releases
   - Version tagging
   - Release notes generation
   - Package publishing

### Environment Setup

- **Development**: Local development environment
- **Testing**: Automated testing environment
- **Staging**: Pre-production testing
- **Production**: Live deployment

## Performance Monitoring

### Key Metrics

1. **Processing Time**: <60 seconds per paper
2. **Reference Removal**: <3 seconds per paper
3. **OCR Accuracy**: >95%
4. **Reference Removal Accuracy**: >99%
5. **Extraction Accuracy**: >90%

### Monitoring Tools

- Performance benchmarking scripts
- Accuracy validation tests
- Memory usage monitoring
- Error rate tracking

### Performance Alerts

- Processing time exceeding thresholds
- Accuracy dropping below targets
- Memory usage spikes
- Error rate increases

## Release Management

### Release Process

1. **Feature Freeze**: No new features
2. **Testing**: Comprehensive testing phase
3. **Documentation**: Update all documentation
4. **Release Branch**: Create release branch
5. **Final Testing**: Production-like testing
6. **Tagging**: Version tagging and release notes
7. **Deployment**: Deploy to production

### Version Scheme

We use Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Notes

Each release includes:
- New features
- Bug fixes
- Performance improvements
- Breaking changes
- Migration guides

## Development Commands Quick Reference

### Setup and Environment
```bash
# Setup development environment
/setup-dev

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests
/test-module all

# Run specific module tests
/test-module reference_cleaner

# Check performance
/check-performance

# Test reference removal
/check-refs data/sample.pdf
```

### Development
```bash
# Run single paper processing
uv run python src/arxiv_extractor.py --id 2301.12345

# Run batch processing
uv run python src/arxiv_extractor.py --batch paper_list.txt

# Full pipeline test
/run-pipeline 2301.12345
```

### GitHub Integration
```bash
# Create issue
/create-issue bug "Description"

# Request review
/review

# Check status
/status
```

## Best Practices

1. **Always create issues** before starting work
2. **Use descriptive branch names** with issue numbers
3. **Write comprehensive commit messages**
4. **Test thoroughly** before creating PRs
5. **Document changes** in PR descriptions
6. **Monitor performance** for critical changes
7. **Review code** carefully and constructively
8. **Keep PRs small** and focused
9. **Update documentation** with code changes
10. **Follow security best practices**

## Troubleshooting

### Common Issues

1. **Test Failures**: Check environment setup and dependencies
2. **Performance Degradation**: Run benchmarks and profiling
3. **OCR Accuracy Issues**: Validate input data and model configuration
4. **Reference Removal Problems**: Check pattern matching and validation
5. **Memory Issues**: Monitor memory usage and optimize batch sizes

### Getting Help

- Use `/help` for Claude Code assistance
- Check issue templates for reporting problems
- Review documentation and guides
- Contact team members for support

---

This workflow guide ensures consistent, efficient development while maintaining high code quality and project standards. Regular updates to this guide will incorporate lessons learned and process improvements.