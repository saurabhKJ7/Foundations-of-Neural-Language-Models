import json
import openai
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv

class QuestionAsker:
    def __init__(self, kb_path: str = "kb.json"):
        """Initialize with knowledge base path"""
        # Load environment variables from .env file
        load_dotenv()
        
        self.kb_path = kb_path
        self.kb_data = self.load_knowledge_base()
        
        # Set up OpenAI client (loads from OPENAI_API_KEY environment variable)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_knowledge_base(self) -> Dict:
        """Load the knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Knowledge base file {self.kb_path} not found!")
            return {"knowledge_base": []}
    
    def get_kb_questions(self) -> List[str]:
        """Extract questions from knowledge base"""
        return [item["question"] for item in self.kb_data["knowledge_base"]]
    
    def get_edge_case_questions(self) -> List[str]:
        """Define 5 edge case questions not in KB"""
        return [
            "What is the population of Mars?",
            "Who invented the time machine?",
            "What is the capital of Atlantis?",
            "How many legs does a unicorn have?",
            "What color is invisible light?"
        ]
    
    def ask_model(self, question: str, model: str = "gpt-3.5-turbo") -> str:
        """Ask the language model a question"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer questions directly and concisely."},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask_all_questions(self) -> List[Tuple[str, str, str]]:
        """Ask all KB questions plus edge cases"""
        results = []
        
        # Ask KB questions
        kb_questions = self.get_kb_questions()
        print(f"Asking {len(kb_questions)} knowledge base questions...")
        
        for i, question in enumerate(kb_questions, 1):
            print(f"[{i}/{len(kb_questions)}] {question}")
            answer = self.ask_model(question)
            results.append((question, answer, "KB"))
            print(f"Answer: {answer}\n")
        
        # Ask edge case questions
        edge_questions = self.get_edge_case_questions()
        print(f"Asking {len(edge_questions)} edge case questions...")
        
        for i, question in enumerate(edge_questions, 1):
            print(f"[{i}/{len(edge_questions)}] {question}")
            answer = self.ask_model(question)
            results.append((question, answer, "EDGE"))
            print(f"Answer: {answer}\n")
        
        return results
    
    def save_results(self, results: List[Tuple[str, str, str]], filename: str = "model_responses.json"):
        """Save results to JSON file"""
        formatted_results = []
        for question, answer, question_type in results:
            formatted_results.append({
                "question": question,
                "model_answer": answer,
                "type": question_type
            })
        
        with open(filename, 'w') as f:
            json.dump({"responses": formatted_results}, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    """Main function to run the question asking process"""
    try:
        asker = QuestionAsker()
        
        # Ask all questions
        results = asker.ask_all_questions()
        
        # Save results
        asker.save_results(results)
        
        print(f"Complete! Asked {len(results)} questions total.")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("2. Or set environment variable: export OPENAI_API_KEY='your-api-key'")
        return

if __name__ == "__main__":
    main()