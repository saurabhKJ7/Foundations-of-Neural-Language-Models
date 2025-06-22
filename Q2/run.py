#!/usr/bin/env python3
"""
Hallucination Detection & Guardrails System
Main orchestration script that runs the complete pipeline
"""

import json
import os
import sys
from dotenv import load_dotenv
from ask_model import QuestionAsker
from validator import HallucinationValidator

def setup_environment():
    """Check if required environment variables are set"""
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key using one of these methods:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("2. Set environment variable: export OPENAI_API_KEY='your-api-key'")
        print("3. Copy .env.template to .env and add your key")
        return False
    return True

def run_pipeline():
    """Run the complete hallucination detection pipeline"""
    print("🧠 Hallucination Detection & Guardrails System")
    print("=" * 50)
    
    # Step 1: Check environment
    if not setup_environment():
        return False
    
    # Step 2: Initialize components
    print("\n📚 Loading Knowledge Base...")
    asker = QuestionAsker("kb.json")
    validator = HallucinationValidator("kb.json")
    
    # Step 3: Ask questions to model
    print("\n🤖 Querying Language Model...")
    results = asker.ask_all_questions()
    asker.save_results(results, "model_responses.json")
    
    # Step 4: Validate responses
    print("\n🔍 Validating Responses...")
    validation_results = validator.validate_responses("model_responses.json")
    validator.save_validation_results(validation_results, "validation_results.json")
    validator.print_summary(validation_results)
    
    # Step 5: Handle retries
    retry_questions = validator.get_retry_questions(validation_results)
    if retry_questions:
        print(f"\n🔄 Processing {len(retry_questions)} retry questions...")
        retry_results = []
        
        for question in retry_questions:
            print(f"Retrying: {question}")
            new_answer = asker.ask_model(question)
            status, message, correct = validator.validate_answer(question, new_answer)
            
            retry_results.append({
                "question": question,
                "retry_answer": new_answer,
                "retry_status": status,
                "retry_message": message,
                "correct_answer": correct
            })
        
        # Save retry results
        with open("retry_results.json", 'w') as f:
            json.dump({"retry_results": retry_results}, f, indent=2)
        
        print(f"✅ Retry results saved to retry_results.json")
    
    print("\n🎯 Pipeline Complete!")
    return True

def generate_summary():
    """Generate final summary report"""
    try:
        # Load validation results
        with open("validation_results.json", 'r') as f:
            validation_data = json.load(f)
        
        results = validation_data["validation_results"]
        
        # Generate summary
        summary = {
            "total_questions": len(results),
            "kb_questions": len([r for r in results if r["question_type"] == "KB"]),
            "edge_questions": len([r for r in results if r["question_type"] == "EDGE"]),
            "passed": len([r for r in results if r["validation_status"] == "PASS"]),
            "kb_mismatches": len([r for r in results if "differs from KB" in r["validation_message"]]),
            "out_of_domain": len([r for r in results if "out-of-domain" in r["validation_message"]])
        }
        
        summary["pass_rate"] = summary["passed"] / summary["total_questions"] * 100
        
        # Write summary
        with open("summary.md", 'w') as f:
            f.write(f"""# Hallucination Detection Results Summary

## Overview
- **Total Questions Asked**: {summary['total_questions']}
- **KB Questions**: {summary['kb_questions']}
- **Edge Case Questions**: {summary['edge_questions']}
- **Overall Pass Rate**: {summary['pass_rate']:.1f}%

## Validation Results
- ✅ **Passed**: {summary['passed']} questions
- ❌ **KB Mismatches**: {summary['kb_mismatches']} questions
- 🚫 **Out-of-Domain**: {summary['out_of_domain']} questions

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
""")
        
        print("📄 Summary report generated: summary.md")
        
    except FileNotFoundError:
        print("❌ Could not generate summary - validation results not found")

def main():
    """Main entry point"""
    try:
        # Run the pipeline
        success = run_pipeline()
        
        if success:
            # Generate summary
            generate_summary()
            
            print("\n🎉 All done! Check the generated files:")
            print("   - kb.json (knowledge base)")
            print("   - model_responses.json (model answers)")
            print("   - validation_results.json (validation details)")
            print("   - retry_results.json (retry attempts)")
            print("   - summary.md (final report)")
        else:
            print("❌ Pipeline failed to complete")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()