import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    def __init__(self, api_key: str, use_mongodb: bool = True):
        self.api_key = api_key
        self.agent = GSM8KAgent(llm_client=LLMClient(api_key=self.api_key))

        # Initialize MongoDB storage if requested
        self.mongo_storage = None
        if use_mongodb:
            try:
                self.mongo_storage = MongoDBStorage()
                print("MongoDB storage initialized")
            except Exception as e:
                print(f"WARNING: Could not initialize MongoDB storage: {e}")
                print("Continuing without MongoDB storage...")

        self.causal_flow = CausalFlow(api_key=self.api_key, mongo_storage=self.mongo_storage)
        self.data_loader = GSM8KDataLoader()
        self.results = []

    def run_experiment(
        self,
        num_rows: int = 5,
        output_dir: str = "gsm8k_results"
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        data = self.data_loader.load_data(num_rows)
        print(f"\nRunning experiment on {len(data)} problems")

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

            trace_file = os.path.join(output_dir, f"trace_{i}.json")
            result['trace'].to_json(trace_file)
            problem_result['trace_file'] = trace_file

            # Save passing traces to MongoDB
            if result['success'] and self.mongo_storage:
                try:
                    self.mongo_storage.save_passing_trace(
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
                    metrics_file = os.path.join(output_dir, f"metrics_{i}.json")
                    analysis = self.causal_flow.analyze_trace(
                        result['trace'],
                        skip_repair=False,
                        metrics_output_file=metrics_file,
                        problem_id=i
                    )
                    report_file = os.path.join(output_dir, f"analysis_{i}.txt")
                    self.causal_flow.generate_full_report(report_file)

                    results_file = os.path.join(output_dir, f"analysis_{i}.json")
                    self.causal_flow.export_results(results_file)

                    causal_steps = []
                    if analysis:
                        if 'multi_agent_critique' in analysis and 'consensus_steps' in analysis['multi_agent_critique']:
                            causal_steps = [s['step_id'] for s in analysis['multi_agent_critique']['consensus_steps']]
                        elif 'causal_attribution' in analysis:
                            causal_steps = analysis['causal_attribution'].get('causal_steps', [])

                    problem_result['causal_analysis'] = {
                        'report_file': report_file,
                        'results_file': results_file,
                        'causal_steps': causal_steps
                    }

                    stats['analyzed'] += 1
                    print(f"  Analysis saved to {report_file}")

                except Exception as e:
                    print(f"  Error during analysis: {e}")
                    problem_result['causal_analysis'] = {'error': str(e)}

            if result['success']:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1

            stats['results'].append(problem_result)

        # Calculate stats for non-skipped problems
        attempted = stats['total'] - stats['skipped']
        stats['accuracy'] = stats['correct'] / attempted if attempted > 0 else 0
        stats['error_rate'] = stats['incorrect'] / attempted if attempted > 0 else 0
        stats['skip_rate'] = stats['skipped'] / stats['total'] if stats['total'] > 0 else 0
        stats['analysis_coverage'] = stats['analyzed'] / stats['incorrect'] if stats['incorrect'] > 0 else 0

        summary_file = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Summary: {summary_file}")

        # Print MongoDB statistics
        if self.mongo_storage:
            mongo_stats = self.mongo_storage.get_statistics()
            print(f"\nMongoDB Statistics:")
            print(f"  Total passing traces: {mongo_stats['total_passing_traces']}")
            print(f"  Total failing traces: {mongo_stats['total_failing_traces']}")
            print(f"  Total traces in DB: {mongo_stats['total_traces']}")

        return stats

    def generate_benefits_report(self, output_dir: str):
        summary_file = os.path.join(output_dir, "experiment_summary.json")

        if not os.path.exists(summary_file):
            print("No summary file found. Run experiment first.")
            return

        with open(summary_file, 'r') as f:
            stats = json.load(f)

        report = []
        report.append("=" * 80)
        report.append("CausalFlow Report for GSM8K")
        report.append("=" * 80)
        report.append("")

        report.append("## Overview")
        report.append(f"Total problems attempted: {stats['total']}")
        report.append(f"Accuracy: {stats['accuracy']*100:.1f}%")
        report.append(f"Failures: {stats['incorrect']}")
        report.append(f"Failures analyzed by CausalFlow: {stats['analyzed']}")
        report.append("")

        report.append(f"CausalFlow automatically identified the root causes of {stats['analyzed']} failures,")
        report.append("")

        report.append("## Failures Analyzed")
        report.append("")

        for i, result in enumerate(stats['results']):
            if not result['success'] and 'causal_analysis' in result:
                report.append(f"### Problem {result['problem_id'] + 1}")
                report.append(f"Question: {result['question']}")
                report.append(f"Expected: {result['gold_answer']}")
                report.append(f"Got: {result['agent_answer']}")

                if 'causal_steps' in result['causal_analysis']:
                    causal_steps = result['causal_analysis']['causal_steps']
                    report.append(f"Causal steps identified: {len(causal_steps)}")
                    report.append(f"See detailed analysis: {result['causal_analysis']['report_file']}")

                report.append("")

        report_file = os.path.join(output_dir, "causalflow_benefits.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"\nReport saved to: {report_file}")
        print('\n'.join(report))


def main():

    load_dotenv()
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_SECRET_KEY not found in .env file")
        return

    experiment = GSM8KExperiment(api_key=api_key)
    num_rows = 11
    stats = experiment.run_experiment(
        num_rows=num_rows,
        output_dir=f"gsm8k_results_{num_rows}"
    )

    if stats['analyzed'] > 0:
        experiment.generate_benefits_report(f"gsm8k_results_{num_rows}")

if __name__ == "__main__":
    main()
