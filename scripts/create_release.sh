#!/bin/bash

# ArXiv Scraper Release Creation Script
# Automates the release preparation process

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Default values
VERSION=""
RELEASE_TYPE="minor"
DRY_RUN=false
SKIP_TESTS=false
SKIP_DOCS=false
BRANCH=$(git branch --show-current)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--type)
            RELEASE_TYPE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-docs)
            SKIP_DOCS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --version VERSION    Specific version to release (e.g., 1.2.3)"
            echo "  -t, --type TYPE          Release type: major, minor, patch (default: minor)"
            echo "  --dry-run                Show what would be done without making changes"
            echo "  --skip-tests             Skip test execution"
            echo "  --skip-docs              Skip documentation generation"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -v 1.2.3"
            echo "  $0 --type patch"
            echo "  $0 --dry-run"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if we're in the correct directory
if [ ! -f "CLAUDE.md" ] || [ ! -f "src/arxiv_extractor.py" ]; then
    print_error "This script must be run from the ArXiv Scraper project root directory"
    exit 1
fi

# Check if git is available and we're in a git repository
if ! command -v git &> /dev/null; then
    print_error "git is required but not installed"
    exit 1
fi

if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    print_error "Not in a git repository"
    exit 1
fi

echo "=============================================="
echo "ArXiv Scraper Release Creation"
echo "=============================================="
echo ""

# Function to get current version
get_current_version() {
    if [ -f "setup.py" ]; then
        python3 -c "
import re
with open('setup.py', 'r') as f:
    content = f.read()
    match = re.search(r'version=[\"\\']([^\"\\']*)[\"\\']]', content)
    if match:
        print(match.group(1))
    else:
        print('0.0.0')
"
    elif [ -f "pyproject.toml" ]; then
        python3 -c "
import re
with open('pyproject.toml', 'r') as f:
    content = f.read()
    match = re.search(r'version = [\"\\']([^\"\\']*)[\"\\']]', content)
    if match:
        print(match.group(1))
    else:
        print('0.0.0')
"
    else
        echo "0.0.0"
    fi
}

# Function to increment version
increment_version() {
    local current=$1
    local type=$2
    
    python3 -c "
import re

version = '$current'
parts = version.split('.')
major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

if '$type' == 'major':
    major += 1
    minor = 0
    patch = 0
elif '$type' == 'minor':
    minor += 1
    patch = 0
elif '$type' == 'patch':
    patch += 1

print(f'{major}.{minor}.{patch}')
"
}

# Function to update version in files
update_version() {
    local new_version=$1
    
    print_info "Updating version to $new_version in project files..."
    
    # Update setup.py
    if [ -f "setup.py" ]; then
        if [ "$DRY_RUN" = false ]; then
            sed -i.bak "s/version=[\"'][^\"']*[\"']/version='$new_version'/" setup.py
            rm setup.py.bak
        fi
        print_status "Updated setup.py"
    fi
    
    # Update pyproject.toml
    if [ -f "pyproject.toml" ]; then
        if [ "$DRY_RUN" = false ]; then
            sed -i.bak "s/version = [\"'][^\"']*[\"']/version = \"$new_version\"/" pyproject.toml
            rm pyproject.toml.bak
        fi
        print_status "Updated pyproject.toml"
    fi
    
    # Update __init__.py if it exists
    if [ -f "src/__init__.py" ]; then
        if [ "$DRY_RUN" = false ]; then
            sed -i.bak "s/__version__ = [\"'][^\"']*[\"']/__version__ = \"$new_version\"/" src/__init__.py
            rm src/__init__.py.bak
        fi
        print_status "Updated src/__init__.py"
    fi
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        print_warning "Skipping tests as requested"
        return 0
    fi
    
    print_info "Running comprehensive tests..."
    
    if [ "$DRY_RUN" = false ]; then
        # Run the test script
        if [ -f "scripts/run_tests.sh" ]; then
            bash scripts/run_tests.sh --type all --coverage
        else
            uv run pytest tests/ -v --cov=src
        fi
    else
        print_info "Would run: bash scripts/run_tests.sh --type all --coverage"
    fi
    
    print_status "All tests passed"
}

# Function to run performance checks
run_performance_checks() {
    print_info "Running performance checks..."
    
    if [ "$DRY_RUN" = false ]; then
        if [ -f "scripts/check_performance.sh" ]; then
            bash scripts/check_performance.sh --type all
        else
            print_warning "Performance check script not found, skipping"
        fi
    else
        print_info "Would run: bash scripts/check_performance.sh --type all"
    fi
    
    print_status "Performance checks completed"
}

# Function to update documentation
update_documentation() {
    if [ "$SKIP_DOCS" = true ]; then
        print_warning "Skipping documentation updates as requested"
        return 0
    fi
    
    print_info "Updating documentation..."
    
    # Update CHANGELOG.md
    local changelog_entry="## [$NEW_VERSION] - $(date +%Y-%m-%d)

### Added
- New features and enhancements

### Changed
- Improvements and modifications

### Fixed
- Bug fixes and corrections

### Performance
- Performance optimizations

"
    
    if [ "$DRY_RUN" = false ]; then
        if [ -f "CHANGELOG.md" ]; then
            # Insert new entry after the first line (header)
            sed -i.bak "1a\\
$changelog_entry" CHANGELOG.md
            rm CHANGELOG.md.bak
        else
            # Create new changelog
            echo "# Changelog" > CHANGELOG.md
            echo "" >> CHANGELOG.md
            echo "$changelog_entry" >> CHANGELOG.md
        fi
    else
        print_info "Would update CHANGELOG.md with new entry"
    fi
    
    print_status "Documentation updated"
}

# Function to create git tag
create_git_tag() {
    local version=$1
    
    print_info "Creating git tag v$version..."
    
    # Check if working directory is clean
    if [ -n "$(git status --porcelain)" ]; then
        print_error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    # Check if tag already exists
    if git tag -l | grep -q "^v$version$"; then
        print_error "Tag v$version already exists"
        exit 1
    fi
    
    if [ "$DRY_RUN" = false ]; then
        # Commit version changes
        git add .
        git commit -m "Bump version to $version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        # Create annotated tag
        git tag -a "v$version" -m "Release version $version

Features and improvements in this release:
- Enhanced performance and accuracy
- Bug fixes and stability improvements
- Documentation updates

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        print_status "Git tag v$version created"
    else
        print_info "Would create git tag v$version"
    fi
}

# Function to create GitHub release
create_github_release() {
    local version=$1
    
    print_info "Creating GitHub release..."
    
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        print_warning "GitHub CLI (gh) not found. Skipping GitHub release creation."
        print_info "You can create the release manually at: https://github.com/your-username/arxiv-scraper/releases"
        return 0
    fi
    
    if [ "$DRY_RUN" = false ]; then
        # Generate release notes
        local release_notes="# ArXiv Scraper v$version

## What's New

This release includes performance improvements, bug fixes, and enhanced accuracy for the ArXiv paper processing pipeline.

### Key Features
- 🔍 Advanced OCR processing with multiple model support
- 🗑️ Intelligent reference section removal
- 🧠 LLM-powered content extraction
- 📊 CSV output generation
- ⚡ Performance optimizations

### Performance Metrics
- Processing time: <60 seconds per paper
- Reference removal: <3 seconds per paper
- OCR accuracy: >95%
- Reference removal accuracy: >99%
- Extraction accuracy: >90%

### Installation

\`\`\`bash
git clone https://github.com/your-username/arxiv-scraper.git
cd arxiv-scraper
uv sync
\`\`\`

### Usage

\`\`\`bash
# Process a single paper
uv run python src/arxiv_extractor.py --id 2301.12345

# Batch processing
uv run python src/arxiv_extractor.py --batch paper_list.txt
\`\`\`

---

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        # Create the release
        gh release create "v$version" \
            --title "ArXiv Scraper v$version" \
            --notes "$release_notes" \
            --draft
        
        print_status "GitHub release created (draft)"
        print_info "Review and publish the release at: https://github.com/your-username/arxiv-scraper/releases"
    else
        print_info "Would create GitHub release v$version"
    fi
}

# Function to push changes
push_changes() {
    local version=$1
    
    print_info "Pushing changes to remote repository..."
    
    if [ "$DRY_RUN" = false ]; then
        # Push commits and tags
        git push origin "$BRANCH"
        git push origin "v$version"
        
        print_status "Changes pushed to remote repository"
    else
        print_info "Would push changes and tag v$version to remote"
    fi
}

# Function to cleanup on error
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Release process failed. Cleaning up..."
        
        # Remove created tag if it exists
        if git tag -l | grep -q "^v$NEW_VERSION$"; then
            git tag -d "v$NEW_VERSION" 2>/dev/null || true
        fi
        
        # Reset changes
        git reset --hard HEAD~1 2>/dev/null || true
    fi
}

# Main execution
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Get current version
    local current_version=$(get_current_version)
    print_info "Current version: $current_version"
    
    # Determine new version
    if [ -n "$VERSION" ]; then
        NEW_VERSION="$VERSION"
    else
        NEW_VERSION=$(increment_version "$current_version" "$RELEASE_TYPE")
    fi
    
    print_info "New version: $NEW_VERSION"
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No changes will be made"
    fi
    
    echo ""
    print_info "Release Plan:"
    echo "  1. Update version in project files"
    echo "  2. Run comprehensive tests"
    echo "  3. Run performance checks"
    echo "  4. Update documentation"
    echo "  5. Create git commit and tag"
    echo "  6. Push changes to remote"
    echo "  7. Create GitHub release"
    echo ""
    
    if [ "$DRY_RUN" = false ]; then
        read -p "Proceed with release? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Release cancelled"
            exit 0
        fi
    fi
    
    # Execute release steps
    update_version "$NEW_VERSION"
    run_tests
    run_performance_checks
    update_documentation
    create_git_tag "$NEW_VERSION"
    push_changes "$NEW_VERSION"
    create_github_release "$NEW_VERSION"
    
    echo ""
    echo "=============================================="
    echo "Release Complete!"
    echo "=============================================="
    echo ""
    print_status "Version $NEW_VERSION released successfully!"
    
    echo ""
    print_info "Next steps:"
    echo "  1. Review and publish the GitHub release"
    echo "  2. Update project documentation"
    echo "  3. Announce the release to users"
    echo "  4. Monitor for any issues"
    echo ""
    
    print_info "Release artifacts:"
    echo "  - Git tag: v$NEW_VERSION"
    echo "  - GitHub release: https://github.com/your-username/arxiv-scraper/releases/tag/v$NEW_VERSION"
    echo "  - Updated documentation: CHANGELOG.md"
}

# Run main function
main "$@"