# Hallucination Detection Results Summary

## Overview
- **Total Questions Asked**: 15
- **KB Questions**: 10
- **Edge Case Questions**: 5
- **Overall Pass Rate**: 53.3%

## Validation Results
- ✅ **Passed**: 8 questions
- ❌ **KB Mismatches**: 2 questions
- 🚫 **Out-of-Domain**: 5 questions

## Files Generated
- `model_responses.json` - Raw model responses
- `validation_results.json` - Detailed validation results
- `retry_results.json` - Results from retry attempts
- `summary.md` - This summary report

## System Behavior
- Questions in KB but with wrong answers → **RETRY: answer differs from KB**
- Questions not in KB → **RETRY: out-of-domain**
- Matching answers → **PASS**
- Each failed question gets one retry attempt
