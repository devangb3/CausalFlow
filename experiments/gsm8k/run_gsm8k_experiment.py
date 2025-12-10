import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from gsm8k_agent import GSM8KAgent
from causal_flow import CausalFlow
from llm_client import LLMClient
from math_reexecutor import MathReexecutor
from mongodb_storage import MongoDBStorage
from datasets import load_dataset

class GSM8KDataLoader:
    def __init__(self):
        self.reexecutor = MathReexecutor()

    def load_data(self, num_rows: Optional[int] = None) -> List[Dict[str, str]]:
        try:
            dataset = load_dataset('gsm8k', 'main', split='test')
            data = [{'question': item['question'], 'answer': item['answer']} for item in dataset]
            print(f"Loaded {len(data)} examples from HuggingFace")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            raise e

        # Limit to num_rows if specified
        if num_rows is not None:
            data = data[:num_rows]

        return data

    def extract_gold_answer(self, answer_text: str) -> str:
        num = self.reexecutor.extract_number(answer_text)
        return str(num) if num is not None else answer_text

class GSM8KExperiment:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.agent = GSM8KAgent(llm_client=LLMClient(api_key=self.api_key))

        self.mongo_storage = None
        try:
            self.mongo_storage = MongoDBStorage()
        except Exception as e:
            raise Exception(f"Could not initialize MongoDB storage: {e}")

        self.causal_flow = CausalFlow(api_key=self.api_key, mongo_storage=self.mongo_storage)

        self.data_loader = GSM8KDataLoader()

    def run_experiment(
        self,
        num_rows: int = 5,
    ):
        data = self.data_loader.load_data(num_rows)
        print(f"\nRunning experiment on {len(data)} problems")

        run_id = self.mongo_storage.create_run(
            experiment_name="GSM8K",
            num_problems=len(data)
        )

        stats = {
            'total': len(data),
            'correct': 0,
            'incorrect': 0,
            'skipped': 0,
            'analyzed': 0,
            'results': []
        }

        for i, item in enumerate(tqdm(data, desc="Solving problems")):
            question = item['question']
            gold_answer = self.data_loader.extract_gold_answer(item['answer'])

            print(f"\n{'='*70}")
            print(f"Problem {i+1}/{len(data)}")
            print(f"Question: {question}")
            print(f"Gold Answer: {gold_answer}")

            result = self.agent.solve(question, gold_answer)

            if result.get('error'):
                print(f"ERROR: Failed to parse response - {result['error']}")
                print(f"Skipping this problem")
                stats['skipped'] += 1

                problem_result = {
                    'problem_id': i,
                    'question': question,
                    'gold_answer': gold_answer,
                    'agent_answer': None,
                    'success': False,
                    'skipped': True,
                    'error': result['error']
                }
                stats['results'].append(problem_result)
                continue

            print(f"Agent Answer: {result['answer']}")
            print(f"Success: {result['success']}")

            problem_result = {
                'problem_id': i,
                'question': question,
                'gold_answer': gold_answer,
                'agent_answer': result['answer'],
                'success': result['success'],
                'skipped': False,
                'num_steps': len(result['trace'].steps)
            }

            if result['success']:
                try:
                    self.mongo_storage.add_passing_trace(
                        run_id=run_id,
                        trace_data=result['trace'].to_json(),
                        problem_id=i,
                        problem_statement=question,
                        gold_answer=gold_answer,
                        final_answer=result['answer']
                    )
                except Exception as e:
                    print(f"  Error saving passing trace to MongoDB: {e}")

            # Analyze failing traces with CausalFlow
            if not result['success']:
                try:
                    analysis = self.causal_flow.analyze_trace(result['trace'])

                    stats['analyzed'] += 1
                    self.mongo_storage.add_failing_trace(
                        run_id=run_id,
                        trace_data=result['trace'].to_json(),
                        problem_id=i,
                        problem_statement=question,
                        gold_answer=gold_answer,
                        final_answer=result['answer'],
                        analysis_results=analysis,
                        metrics=analysis['metrics']
                    )
                except Exception as e:
                    print(f"  Error during analysis: {e}")
                    
            if result['success']:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1
        
        run_stats = self.mongo_storage.get_run_statistics(run_id)
        print(f"\nMongoDB Statistics for this run:")
        print(f"  Run ID: {run_stats['run_id']}")
        print(f"  Total traces: {run_stats['total_traces']}")
        print(f"  Passing traces: {run_stats['passing_traces']}")
        print(f"  Failing traces: {run_stats['failing_traces']}")
        print(f"  Accuracy: {run_stats['accuracy']:.2%}")

def main():

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        return

    experiment = GSM8KExperiment(api_key=api_key)
    num_rows = 250

    experiment.run_experiment(num_rows=num_rows)

if __name__ == "__main__":
    main()
