# Hallucination Detection & Guardrails System

A Python-based system to detect hallucinations in language model responses by verifying answers against a knowledge base using string matching validation.

## 🎯 Overview

This system implements a simple but effective approach to detect when language models produce incorrect or fabricated information by:

1. **Knowledge Base Verification**: Comparing model answers against a curated knowledge base
2. **String Matching Validation**: Using fuzzy string matching to detect answer discrepancies
3. **Automatic Retry Logic**: Re-querying the model when hallucinations are detected
4. **Out-of-Domain Detection**: Identifying questions outside the knowledge base scope

## 📁 Project Structure

```
Q2/
├── kb.json                 # Knowledge base with 10 factual Q-A pairs
├── ask_model.py           # Script to query language model
├── validator.py           # Hallucination detection validator
├── run.py                 # Main orchestration script
├── README.md              # This documentation
└── Generated Files:
    ├── model_responses.json    # Raw model responses
    ├── validation_results.json # Validation outcomes
    ├── retry_results.json      # Retry attempt results
    └── summary.md             # Final report
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.7+** with required packages:
   ```bash
   pip install openai
   ```

2. **OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

### Running the System

Execute the complete pipeline:
```bash
python run.py
```

Or run components individually:
```bash
# Step 1: Query the model
python ask_model.py

# Step 2: Validate responses
python validator.py
```

## 🧠 System Components

### 1. Knowledge Base (`kb.json`)

Contains 10 factual question-answer pairs covering various domains:
- Geography (capitals, continents)
- Science (chemistry, biology, physics)
- Literature (authors, works)
- History (dates, events)
- General knowledge (animals, shapes)

### 2. Question Asker (`ask_model.py`)

**Purpose**: Query language model with KB questions + 5 edge cases

**Features**:
- Loads questions from knowledge base
- Includes 5 deliberately tricky/impossible questions
- Uses OpenAI GPT-3.5-turbo for responses
- Saves all responses for validation

**Edge Case Questions**:
- "What is the population of Mars?"
- "Who invented the time machine?"
- "What is the capital of Atlantis?"
- "How many legs does a unicorn have?"
- "What color is invisible light?"

### 3. Validator (`validator.py`)

**Purpose**: Detect hallucinations using string matching

**Validation Logic**:
```
If question in KB:
    If answer matches KB → PASS
    If answer differs → RETRY: answer differs from KB
Else:
    → RETRY: out-of-domain
```

**String Matching Methods**:
- Exact match (after normalization)
- Substring containment
- Fuzzy similarity (SequenceMatcher, threshold=0.8)

**Text Normalization**:
- Convert to lowercase
- Remove extra whitespace
- Strip punctuation
- Handle common variations

### 4. Main Orchestrator (`run.py`)

**Purpose**: Execute complete pipeline with retry logic

**Pipeline Steps**:
1. Environment setup (API key check)
2. Load knowledge base
3. Query model with all questions
4. Validate responses against KB
5. Retry failed questions (once)
6. Generate comprehensive report

## 📊 Output Files

### `model_responses.json`
Raw responses from the language model:
```json
{
  "responses": [
    {
      "question": "What is the capital of France?",
      "model_answer": "Paris",
      "type": "KB"
    }
  ]
}
```

### `validation_results.json`
Detailed validation outcomes:
```json
{
  "validation_results": [
    {
      "question": "What is the capital of France?",
      "model_answer": "Paris",
      "validation_status": "PASS",
      "validation_message": "answer matches KB",
      "correct_answer": "paris",
      "needs_retry": false
    }
  ]
}
```

### `summary.md`
Executive summary with statistics and system behavior analysis.

## 🎛️ Configuration

### Validation Sensitivity
Adjust string matching threshold in `validator.py`:
```python
def is_string_match(self, answer1: str, answer2: str, threshold: float = 0.8):
```

### Model Parameters
Modify model settings in `ask_model.py`:
```python
response = self.client.chat.completions.create(
    model="gpt-3.5-turbo",        # Model selection
    max_tokens=150,               # Response length
    temperature=0.1               # Creativity level
)
```

## 🔍 Validation Examples

### ✅ PASS Cases
- **Question**: "What is the capital of France?"
- **Model**: "Paris"
- **KB**: "Paris"
- **Result**: PASS (exact match)

### ❌ KB Mismatch
- **Question**: "How many continents are there?"
- **Model**: "There are 5 continents"
- **KB**: "7"
- **Result**: RETRY: answer differs from KB

### 🚫 Out-of-Domain
- **Question**: "What is the population of Mars?"
- **Model**: "Mars has no permanent population"
- **KB**: (not found)
- **Result**: RETRY: out-of-domain

## 📈 Performance Metrics

The system tracks:
- **Pass Rate**: Percentage of correctly answered questions
- **KB Coverage**: Questions successfully found in knowledge base
- **Mismatch Rate**: KB questions with incorrect answers
- **Domain Coverage**: Ratio of in-domain vs out-of-domain questions

## 🛡️ Guardrails Implementation

### Retry Logic
- Each failed validation triggers **one retry attempt**
- Retry uses same model parameters
- Results tracked separately for analysis

### Error Handling
- API failures gracefully handled
- Missing files create appropriate defaults
- Invalid responses logged and reported

## 🔧 Troubleshooting

### Common Issues

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY='your-key'
```

**"Knowledge base file not found"**
- Ensure `kb.json` exists in working directory
- Check file permissions

**Low pass rates**
- Adjust string matching threshold
- Review KB answer formatting
- Consider model temperature settings

## 🎯 Use Cases

### Development & Testing
- **Model Evaluation**: Assess factual accuracy
- **Guardrail Testing**: Validate safety mechanisms
- **Regression Testing**: Monitor model performance changes

### Production Applications
- **Content Filtering**: Flag potentially incorrect information
- **User Warnings**: Alert users to uncertain responses
- **Quality Assurance**: Maintain response reliability

## 🚀 Extensions

### Possible Enhancements
1. **Semantic Matching**: Use embeddings instead of string matching
2. **Confidence Scoring**: Probability-based validation
3. **Dynamic KB**: Automatically update knowledge base
4. **Multi-Model Support**: Compare responses across models
5. **Web Integration**: Real-time fact-checking APIs

### Advanced Features
- Machine learning-based hallucination detection
- Domain-specific knowledge bases
- Integration with external fact-checking services
- Real-time monitoring and alerting

## 📝 License

This project is provided as-is for educational and research purposes.