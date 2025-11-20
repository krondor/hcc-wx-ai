# 🚀 Getting Started with watsonx.ai

A quick start guide for Houston Community College students to get up and running with watsonx.ai in 30 minutes.

## 📋 What You'll Need

- [ ] Computer with internet connection
- [ ] IBM Cloud account (free tier available)
- [ ] Python 3.9 or higher installed
- [ ] Basic Python knowledge
- [ ] 30 minutes of time

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- ✅ Set up your watsonx.ai environment
- ✅ Run your first AI model
- ✅ Generate text with foundation models
- ✅ Understand basic prompt engineering

## Step 1: Create IBM Cloud Account (5 minutes)

### 1.1 Sign Up

1. Visit [IBM Cloud Registration](https://cloud.ibm.com/registration)
2. Fill in your information:
   - Email address (use your HCC email)
   - Password (strong password required)
   - First and last name
   - Country/Region
3. Accept terms and conditions
4. Click "Create Account"

### 1.2 Verify Email

1. Check your email inbox
2. Click the verification link
3. Log in to IBM Cloud

**💡 Tip**: Check spam folder if you don't see the email within 5 minutes.

## Step 2: Set Up watsonx.ai Service (5 minutes)

### 2.1 Create watsonx.ai Instance

1. Log in to [IBM Cloud Console](https://cloud.ibm.com)
2. Click "Catalog" in the top menu
3. Search for "watsonx.ai"
4. Click on "watsonx.ai" service
5. Select the **Lite (Free)** plan
6. Click "Create"

### 2.2 Create a Project

1. After service creation, click "Launch watsonx.ai"
2. Click "Projects" in the left sidebar
3. Click "New project"
4. Choose "Create an empty project"
5. Enter project details:
   - **Name**: "HCC Learning Project"
   - **Description**: "My first watsonx.ai project"
6. Click "Create"

### 2.3 Get Your Project ID

1. In your project, click the "Manage" tab
2. Click "General"
3. Find and copy your **Project ID**
4. Save it somewhere safe (you'll need it later)

**📝 Note**: Your Project ID looks like: `12345678-1234-1234-1234-123456789abc`

## Step 3: Get API Credentials (5 minutes)

### 3.1 Create API Key

1. Go to [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys)
2. Click "Create an IBM Cloud API key"
3. Enter a name: "watsonx-learning-key"
4. Add description: "API key for HCC watsonx.ai learning"
5. Click "Create"

### 3.2 Save Your API Key

⚠️ **IMPORTANT**: You can only see your API key once!

1. Click "Download" or copy the key
2. Save it in a secure location
3. **Never share your API key** with anyone
4. **Never commit it to Git** or public repositories

**💡 Tip**: Store it in a password manager or secure note-taking app.

## Step 4: Set Up Your Development Environment (10 minutes)

### 4.1 Install Python

Check if Python is installed:

```bash
python --version
# or
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

**Required**: Python 3.9 or higher

### 4.2 Clone the Repository

```bash
# Clone the HCC watsonx.ai repository
git clone https://github.com/your-org/hcc-wx-ai.git

# Navigate to the directory
cd hcc-wx-ai
```

**Don't have Git?** Download as ZIP from GitHub and extract.

### 4.3 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**✅ Success indicator**: Your terminal prompt should show `(venv)` at the beginning.

### 4.4 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install ibm-watsonx-ai jupyter python-dotenv
```

**⏱️ This may take 2-3 minutes**

### 4.5 Configure Credentials

Create a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your favorite text editor
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

Add your credentials:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

**Save and close** the file.

## Step 5: Run Your First Model (5 minutes)

### 5.1 Create Test Script

Create a file named `test_setup.py`:

```python
"""
Test watsonx.ai Setup
HCC Learning Hub
"""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

print("🔧 Loading credentials...")
credentials = Credentials(
    url=os.getenv('WATSONX_URL'),
    api_key=os.getenv('WATSONX_API_KEY')
)

print("🤖 Initializing model...")
model = ModelInference(
    model_id="ibm/granite-13b-instruct-v2",
    credentials=credentials,
    project_id=os.getenv('WATSONX_PROJECT_ID')
)

print("💬 Generating response...")
prompt = "What is artificial intelligence? Explain in simple terms."

response = model.generate_text(
    prompt=prompt,
    params={
        "max_new_tokens": 100,
        "temperature": 0.7
    }
)

print("\n" + "="*50)
print("PROMPT:", prompt)
print("="*50)
print("RESPONSE:", response)
print("="*50)
print("\n✅ Success! Your watsonx.ai setup is working!")
```

### 5.2 Run the Test

```bash
python test_setup.py
```

**Expected Output**:
```
🔧 Loading credentials...
🤖 Initializing model...
💬 Generating response...
==================================================
PROMPT: What is artificial intelligence? Explain in simple terms.
==================================================
RESPONSE: Artificial intelligence (AI) is the simulation of human intelligence...
==================================================

✅ Success! Your watsonx.ai setup is working!
```

### 5.3 Troubleshooting

**Error: "Invalid API key"**
- Check your API key in `.env` file
- Ensure no extra spaces
- Verify key hasn't expired

**Error: "Project not found"**
- Verify Project ID is correct
- Ensure you have access to the project
- Check you're using the right IBM Cloud account

**Error: "Module not found"**
- Ensure virtual environment is activated
- Run `pip install ibm-watsonx-ai` again

## 🎉 Congratulations!

You've successfully set up watsonx.ai! Here's what you accomplished:

✅ Created IBM Cloud account  
✅ Set up watsonx.ai service  
✅ Got API credentials  
✅ Installed Python environment  
✅ Ran your first AI model  

## 🚀 Next Steps

### Beginner Path (Start Here!)

1. **Explore Foundation Models** (30 min)
   - Open: `notebooks/python_sdk/deployments/foundation_models/`
   - Try: `Use watsonx, and granite-13b-instruct to analyze car rental customer satisfaction.ipynb`

2. **Learn Prompt Engineering** (45 min)
   - Experiment with different prompts
   - Try different temperature settings
   - Compare model outputs

3. **Question Answering** (45 min)
   - Try: `Use watsonx, and Meta llama-3-70b-instruct to answer question about an article.ipynb`

### Launch Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Navigate to the `notebooks/` directory and start exploring!

## 📚 Learning Resources

### Essential Documentation
- [watsonx.ai Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/welcome-main.html?context=wx&audience=wdp)
- [Python SDK Docs](https://ibm.github.io/watsonx-ai-python-sdk/v1.4.7/samples.html)
- [Main README](README.md) - Comprehensive guide

### Quick References

**Common Model IDs:**
```python
# IBM Granite (recommended for beginners)
"ibm/granite-13b-instruct-v2"
"ibm/granite-20b-multilingual"

# Meta Llama
"meta-llama/llama-3-70b-instruct"
"meta-llama/llama-2-70b-chat"

# Code Generation
"codellama/codellama-34b-instruct-hf"

# Mistral
"mistralai/mixtral-8x7b-instruct-v01"
```

**Common Parameters:**
```python
params = {
    "max_new_tokens": 100,      # Maximum response length
    "temperature": 0.7,          # Creativity (0.0-1.0)
    "top_p": 0.9,               # Nucleus sampling
    "top_k": 50,                # Top-k sampling
    "repetition_penalty": 1.1   # Avoid repetition
}
```

## 💡 Tips for Success

### 1. Start Simple
- Begin with basic text generation
- Gradually increase complexity
- Don't try to learn everything at once

### 2. Experiment
- Try different prompts
- Adjust parameters
- Compare model outputs

### 3. Read Error Messages
- Error messages are helpful!
- Google the error if stuck
- Check the troubleshooting section

### 4. Use the Community
- Ask questions in HCC forums
- Share your discoveries
- Help other students

### 5. Practice Regularly
- Spend 30 minutes daily
- Complete one notebook per week
- Build small projects

## 🆘 Getting Help

### If You're Stuck

1. **Check the README** - Comprehensive troubleshooting section
2. **Review Documentation** - IBM's official docs
3. **Ask in Forums** - HCC course forums
4. **Contact Instructor** - Your instructor is here to help
5. **GitHub Issues** - Report bugs or ask questions

### Common Questions

**Q: How much does this cost?**  
A: The free tier is sufficient for learning. You won't be charged unless you exceed free tier limits.

**Q: Can I use this on my personal projects?**  
A: Yes! The skills you learn apply to any watsonx.ai project.

**Q: What if I make a mistake?**  
A: That's how you learn! Experiment freely. You can't break anything.

**Q: How long will it take to learn?**  
A: Basic skills: 4-6 hours. Intermediate: 8-12 hours. Advanced: 15-20 hours.

## 📞 Support

- **HCC Course Forums**: [Link to forums]
- **Instructor Email**: [instructor@hcc.edu]
- **Office Hours**: [Schedule]
- **GitHub Issues**: [Report technical issues](https://github.com/your-org/hcc-wx-ai/issues)

## 🎯 Your Learning Journey

```
You Are Here → 🟢 Beginner → 🟡 Intermediate → 🔴 Advanced → 🏆 Expert
```

**Current Status**: Setup Complete! ✅

**Next Milestone**: Complete first notebook 📓

**Goal**: Build your first AI application 🚀

---

**Ready to start learning?** Head to the [main README](README.md) and choose your learning path!

**Happy Learning! 🎓**