import json
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

class HallucinationValidator:
    def __init__(self, kb_path: str = "kb.json"):
        """Initialize validator with knowledge base"""
        self.kb_path = kb_path
        self.kb_data = self.load_knowledge_base()
        self.kb_qa_map = self.create_qa_mapping()
        
    def load_knowledge_base(self) -> Dict:
        """Load knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Knowledge base file {self.kb_path} not found!")
            return {"knowledge_base": []}
    
    def create_qa_mapping(self) -> Dict[str, str]:
        """Create mapping of questions to correct answers"""
        qa_map = {}
        for item in self.kb_data["knowledge_base"]:
            qa_map[item["question"].lower().strip()] = item["answer"].lower().strip()
        return qa_map
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove common punctuation
        text = re.sub(r'[.,!?;:]', '', text)
        return text
    
    def is_string_match(self, answer1: str, answer2: str, threshold: float = 0.8) -> bool:
        """Check if two answers match using string similarity"""
        norm_answer1 = self.normalize_text(answer1)
        norm_answer2 = self.normalize_text(answer2)
        
        # Exact match
        if norm_answer1 == norm_answer2:
            return True
        
        # Substring match
        if norm_answer1 in norm_answer2 or norm_answer2 in norm_answer1:
            return True
        
        # Similarity match using SequenceMatcher
        similarity = SequenceMatcher(None, norm_answer1, norm_answer2).ratio()
        return similarity >= threshold
    
    def find_kb_question(self, question: str) -> Optional[str]:
        """Find if question exists in KB (case-insensitive)"""
        norm_question = question.lower().strip()
        for kb_question in self.kb_qa_map.keys():
            if self.normalize_text(norm_question) == self.normalize_text(kb_question):
                return kb_question
        return None
    
    def validate_answer(self, question: str, model_answer: str) -> Tuple[str, str, Optional[str]]:
        """
        Validate model answer against KB
        Returns: (status, message, correct_answer_if_available)
        """
        # Check if question is in KB
        kb_question = self.find_kb_question(question)
        
        if kb_question is None:
            return "RETRY", "out-of-domain", None
        
        # Question is in KB, check answer
        correct_answer = self.kb_qa_map[kb_question]
        
        if self.is_string_match(model_answer, correct_answer):
            return "PASS", "answer matches KB", correct_answer
        else:
            return "RETRY", "answer differs from KB", correct_answer
    
    def validate_responses(self, responses_file: str = "model_responses.json") -> List[Dict]:
        """
        Validate all responses from model
        Returns list of validation results
        """
        try:
            with open(responses_file, 'r') as f:
                data = json.load(f)
            responses = data.get("responses", [])
        except FileNotFoundError:
            print(f"Responses file {responses_file} not found!")
            return []
        
        validation_results = []
        
        for response in responses:
            question = response["question"]
            model_answer = response["model_answer"]
            question_type = response.get("type", "UNKNOWN")
            
            status, message, correct_answer = self.validate_answer(question, model_answer)
            
            result = {
                "question": question,
                "model_answer": model_answer,
                "question_type": question_type,
                "validation_status": status,
                "validation_message": message,
                "correct_answer": correct_answer,
                "needs_retry": status == "RETRY"
            }
            
            validation_results.append(result)
            
            # Print validation result
            print(f"Q: {question}")
            print(f"Model Answer: {model_answer}")
            print(f"Status: {status} - {message}")
            if correct_answer:
                print(f"Correct Answer: {correct_answer}")
            print("-" * 50)
        
        return validation_results
    
    def save_validation_results(self, results: List[Dict], filename: str = "validation_results.json"):
        """Save validation results to file"""
        with open(filename, 'w') as f:
            json.dump({"validation_results": results}, f, indent=2)
        print(f"Validation results saved to {filename}")
    
    def get_retry_questions(self, results: List[Dict]) -> List[str]:
        """Extract questions that need retry"""
        return [result["question"] for result in results if result["needs_retry"]]
    
    def print_summary(self, results: List[Dict]):
        """Print validation summary"""
        total = len(results)
        passed = sum(1 for r in results if r["validation_status"] == "PASS")
        retry_kb_mismatch = sum(1 for r in results if r["validation_status"] == "RETRY" and "differs from KB" in r["validation_message"])
        retry_out_of_domain = sum(1 for r in results if r["validation_status"] == "RETRY" and "out-of-domain" in r["validation_message"])
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {total}")
        print(f"Passed: {passed}")
        print(f"Retry - KB Mismatch: {retry_kb_mismatch}")
        print(f"Retry - Out of Domain: {retry_out_of_domain}")
        print(f"Pass Rate: {passed/total*100:.1f}%")
        print("="*60)

def main():
    """Main function to run validation"""
    validator = HallucinationValidator()
    
    # Validate responses
    results = validator.validate_responses()
    
    if results:
        # Save results
        validator.save_validation_results(results)
        
        # Print summary
        validator.print_summary(results)
        
        # Show retry questions
        retry_questions = validator.get_retry_questions(results)
        if retry_questions:
            print(f"\nQuestions needing retry ({len(retry_questions)}):")
            for i, question in enumerate(retry_questions, 1):
                print(f"{i}. {question}")

if __name__ == "__main__":
    main()