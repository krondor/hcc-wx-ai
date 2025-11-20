# Contributing to HCC watsonx.ai Learning Hub

Thank you for your interest in contributing to the Houston Community College watsonx.ai Learning Hub! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- 📓 **Add new notebooks** - Share your learning examples
- 🐛 **Report bugs** - Help us identify issues
- 📚 **Improve documentation** - Enhance explanations and guides
- 💡 **Suggest features** - Propose new learning paths or topics
- 🎨 **Improve examples** - Enhance existing notebooks
- ❓ **Answer questions** - Help other students in discussions

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/hcc-wx-ai.git
cd hcc-wx-ai

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/hcc-wx-ai.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black flake8 pytest
```

### 3. Create a Branch

```bash
# Create a descriptive branch name
git checkout -b feature/add-sentiment-analysis-notebook
# or
git checkout -b fix/correct-typo-in-readme
# or
git checkout -b docs/improve-setup-instructions
```

## 📓 Adding a New Notebook

### Notebook Structure

Every notebook should follow this structure:

```markdown
# Title: Clear, Descriptive Title

## 📋 Overview
Brief description of what this notebook demonstrates (2-3 sentences)

## 🎯 Learning Objectives
After completing this notebook, you will be able to:
- Objective 1
- Objective 2
- Objective 3

## 📚 Prerequisites
- Required knowledge (e.g., "Basic Python", "Understanding of ML concepts")
- Required packages (list any beyond standard requirements.txt)
- Required data (if any)

## ⏱️ Estimated Time
Approximately X minutes/hours

## 🛠️ Setup

### Import Libraries
[Code cell with imports]

### Load Credentials
[Code cell for authentication]

### Initialize Model/Service
[Code cell for setup]

## 📖 Main Content

### Section 1: [Topic]
[Explanation in markdown]
[Code cell]
[Output/Results]

### Section 2: [Topic]
[Explanation in markdown]
[Code cell]
[Output/Results]

## 📊 Results and Analysis
[Visualization and interpretation]

## 🎓 Key Takeaways
- Takeaway 1
- Takeaway 2
- Takeaway 3

## 🚀 Next Steps
Suggestions for further exploration:
- Try modifying X
- Explore Y
- Build upon this with Z

## 📚 Resources
- [Link to relevant documentation]
- [Link to related notebooks]
- [Link to additional reading]

## 🤝 Contributing
Found an issue or have a suggestion? Please open an issue or submit a pull request!
```

### Naming Conventions

**Notebook Files:**
- Use descriptive names: `Use watsonx to [action] [object].ipynb`
- Examples:
  - ✅ `Use watsonx to analyze customer sentiment.ipynb`
  - ✅ `Use AutoAI to predict credit risk.ipynb`
  - ❌ `notebook1.ipynb`
  - ❌ `test.ipynb`

**Directory Structure:**
```
notebooks/
├── python_sdk/
│   ├── deployments/
│   │   ├── foundation_models/
│   │   ├── scikit-learn/
│   │   └── ...
│   └── experiments/
│       ├── autoai/
│       └── ...
```

### Code Style Guidelines

#### Python Code (PEP 8)

```python
# ✅ Good: Clear, documented, error handling
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv

def analyze_sentiment(text: str, model: ModelInference) -> dict:
    """
    Analyze sentiment of given text using watsonx.ai model.
    
    Args:
        text: Input text to analyze
        model: Initialized ModelInference instance
        
    Returns:
        Dictionary with sentiment score and label
        
    Raises:
        ValueError: If text is empty
        Exception: If model inference fails
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        prompt = f"Analyze the sentiment of this text: {text}"
        response = model.generate_text(prompt)
        return parse_sentiment_response(response)
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        raise

# ❌ Bad: Unclear, no documentation, no error handling
def analyze(t, m):
    p = f"Analyze the sentiment of this text: {t}"
    r = m.generate_text(p)
    return parse_sentiment_response(r)
```

#### Markdown Style

```markdown
# ✅ Good: Clear headers, organized content

## Section Title

Brief introduction to the section.

### Subsection

Detailed explanation with:
- Bullet points for lists
- **Bold** for emphasis
- `code` for inline code
- Links to [resources](https://example.com)

### Code Example

\`\`\`python
# Clear, commented code
result = model.generate_text(prompt)
\`\`\`

# ❌ Bad: No structure, unclear

some text here
code without formatting
no organization
```

### Testing Your Notebook

Before submitting, ensure:

1. **Run all cells** from top to bottom (Kernel → Restart & Run All)
2. **Verify outputs** are correct and make sense
3. **Check for errors** - no error messages in output
4. **Test with fresh environment** if possible
5. **Clear outputs** (optional, but recommended for version control)

```bash
# Clear notebook outputs (optional)
jupyter nbconvert --clear-output --inplace your_notebook.ipynb
```

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** - Your bug may already be reported
2. **Try the latest version** - Update packages and retry
3. **Reproduce the bug** - Ensure it's consistent

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run cell '...'
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Screenshots**
If applicable, add screenshots

**Environment**
- OS: [e.g., macOS 13.0, Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- ibm-watsonx-ai version: [e.g., 1.0.0]
- Jupyter version: [e.g., 7.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How would you implement this feature?

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Any other relevant information, mockups, or examples
```

## 📝 Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure you're on your feature branch
git checkout feature/your-feature-name

# Add your changes
git add .

# Commit with descriptive message
git commit -m "Add sentiment analysis notebook with Granite model"
```

### 2. Update Documentation

- Update README.md if adding new notebooks
- Add entry to appropriate section
- Update table of contents if needed

### 3. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to GitHub and navigate to your fork
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template:

```markdown
**Description**
Brief description of changes

**Type of Change**
- [ ] New notebook
- [ ] Bug fix
- [ ] Documentation update
- [ ] Feature enhancement

**Checklist**
- [ ] Code follows style guidelines
- [ ] Notebook runs without errors
- [ ] Documentation updated
- [ ] Self-review completed
- [ ] Comments added for complex code

**Testing**
Describe how you tested your changes

**Screenshots** (if applicable)
Add screenshots of outputs or visualizations

**Related Issues**
Closes #123 (if applicable)
```

### 5. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged
- You'll be added to the contributors list!

## 📋 Code Review Checklist

### For Reviewers

- [ ] Code follows style guidelines
- [ ] Notebook structure is clear and organized
- [ ] All cells run without errors
- [ ] Documentation is clear and complete
- [ ] Learning objectives are well-defined
- [ ] Code includes appropriate error handling
- [ ] Examples are relevant and educational
- [ ] No sensitive information (API keys, passwords) in code

### For Contributors

Before requesting review:
- [ ] I have tested my notebook thoroughly
- [ ] I have added appropriate documentation
- [ ] I have followed the style guidelines
- [ ] I have updated the README if needed
- [ ] My code includes error handling
- [ ] I have added comments for complex logic

## 🎨 Style Guide Summary

### Python
- Follow PEP 8
- Use type hints where appropriate
- Include docstrings for functions
- Add comments for complex logic
- Use meaningful variable names

### Markdown
- Use headers for organization (# ## ###)
- Include code blocks with language specification
- Add links to resources
- Use bullet points for lists
- Bold important terms

### Notebooks
- Clear title and overview
- Learning objectives listed
- Prerequisites specified
- Code cells with explanations
- Results and analysis included
- Next steps suggested

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- HCC course acknowledgments
- GitHub contributors page

## 📞 Getting Help

Need help with your contribution?

- 💬 **GitHub Discussions** - Ask questions
- 📧 **Course Instructors** - Contact for guidance
- 🤝 **HCC AI Community** - Join discussions
- 📚 **Documentation** - Review guides and examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Thank You!

Thank you for contributing to the HCC watsonx.ai Learning Hub! Your contributions help fellow students learn and grow in their AI journey.

---

**Questions?** Open an issue or reach out to the maintainers.

**Happy Contributing! 🚀**