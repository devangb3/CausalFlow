"""
GSM8K Experiment Runner with CausalFlow Analysis

This script demonstrates the benefits of CausalFlow architecture on GSM8K dataset:
1. Loads GSM8K test data (configurable number of rows)
2. Runs the agent on each problem
3. For failed cases, performs causal analysis
4. Generates comprehensive reports showing causal attribution and repairs
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

from gsm8k_agent import GSM8KAgent
from causal_flow import CausalFlow
from llm_client import LLMClient
from math_reexecutor import MathReexecutor


# Sample GSM8K problems for testing when HuggingFace is unavailable
SAMPLE_GSM8K_DATA = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fiber\n#### 3"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 80,000+120,000=$<<80000+120000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": "He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 meters\n#### 540"
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. If Wendi has 20 chickens, how many cups of feed does she need per day?",
        "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\n#### 60"
    },
    {
        "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "answer": "The discount price of one glass is 60/100*5=$<<60/100*5=3>>3.\nIf every second glass is cheaper, that means Kylar is going to buy 16/2=<<16/2=8>>8 cheaper glasses.\nSo for the cheaper glasses, Kylar is going to pay 8*3=$<<8*3=24>>24.\nAnd for the regular-priced glasses, Kylar will pay 8*5=$<<8*5=40>>40.\nSo in total Kylar needs to pay 24+40=$<<24+40=64>>64 for the glasses he wants to buy.\n#### 64"
    },
    {
        "question": "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?",
        "answer": "First figure out how many hours it takes to hike a 12-mile trail at 4 mph by dividing the distance by the speed: 12 miles / 4 mph = <<12/4=3>>3 hours\nNext subtract the time Marissa already spent walking to find out how much time she has left: 3 hours - 1 hour - 1 hour = <<3-1-1=1>>1 hour\nNow figure out how much distance she has left by subtracting the distance she already traveled from the total distance: 12 miles - 4 miles - 2 miles = <<12-4-2=6>>6 miles\nNow divide the remaining distance by the remaining time to find out how fast in miles per hour Marissa has to travel: 6 miles / 1 hour = <<6/1=6>>6 mph\n#### 6"
    },
    {
        "question": "I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?",
        "answer": "There are 15 x 3/5 = <<15*3/5=9>>9 liters of water from the 15 liters pineapple drink.\nAfter 1 liter of orange drink was spilled, there were 10 - 1 = <<10-1=9>>9 liters of orange drink left.\nOut of the 9 liters, 9 x 2/3 = <<9*2/3=6>>6 liters are water.\nThus, there are a total of 9 + 6 = <<9+6=15>>15 liters of water out of the 24 liters.\n#### 15"
    },
]


class GSM8KDataLoader:
    """Loads GSM8K dataset from HuggingFace or uses sample data."""

    def __init__(self):
        self.reexecutor = MathReexecutor()

    def load_data(self, num_rows: Optional[int] = None, use_sample: bool = False) -> List[Dict[str, str]]:
        """
        Load GSM8K test data.

        Args:
            num_rows: Number of rows to load (None for all)
            use_sample: Force use of sample data instead of HuggingFace

        Returns:
            List of dictionaries with 'question' and 'answer' keys
        """
        if use_sample:
            data = SAMPLE_GSM8K_DATA
        else:
            try:
                from datasets import load_dataset
                print("Loading GSM8K dataset from HuggingFace...")
                dataset = load_dataset('gsm8k', 'main', split='test')
                data = [{'question': item['question'], 'answer': item['answer']} for item in dataset]
                print(f"Loaded {len(data)} examples from HuggingFace")
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                print("Using sample data instead...")
                data = SAMPLE_GSM8K_DATA

        # Limit to num_rows if specified
        if num_rows is not None:
            data = data[:num_rows]

        return data

    def extract_gold_answer(self, answer_text: str) -> str:
        """
        Extract numerical answer from GSM8K answer format.

        Args:
            answer_text: Answer text in GSM8K format (with #### prefix)

        Returns:
            Numerical answer as string
        """
        num = self.reexecutor.extract_number(answer_text)
        return str(num) if num is not None else answer_text


class GSM8KExperiment:
    """Runs GSM8K experiments with CausalFlow analysis."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize experiment.

        Args:
            api_key: OpenRouter API key (uses env var if None)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_SECRET_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_SECRET_KEY not found in environment")

        self.agent = GSM8KAgent(llm_client=LLMClient(api_key=self.api_key))
        self.causal_flow = CausalFlow(api_key=self.api_key)
        self.data_loader = GSM8KDataLoader()
        self.results = []

    def run_experiment(
        self,
        num_rows: int = 5,
        use_sample: bool = False,
        analyze_failures: bool = True,
        output_dir: str = "gsm8k_results"
    ) -> Dict[str, Any]:
        """
        Run the GSM8K experiment.

        Args:
            num_rows: Number of problems to solve
            use_sample: Use sample data instead of HuggingFace
            analyze_failures: Whether to run CausalFlow on failures
            output_dir: Directory to save results

        Returns:
            Summary statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        data = self.data_loader.load_data(num_rows, use_sample)
        print(f"\nRunning experiment on {len(data)} problems...")

        # Track statistics
        stats = {
            'total': len(data),
            'correct': 0,
            'incorrect': 0,
            'analyzed': 0,
            'results': []
        }

        # Process each problem
        for i, item in enumerate(tqdm(data, desc="Solving problems")):
            question = item['question']
            gold_answer = self.data_loader.extract_gold_answer(item['answer'])

            print(f"\n{'='*70}")
            print(f"Problem {i+1}/{len(data)}")
            print(f"Question: {question}")
            print(f"Gold Answer: {gold_answer}")

            # Solve the problem
            result = self.agent.solve(question, gold_answer)

            print(f"Agent Answer: {result['answer']}")
            print(f"Success: {result['success']}")

            # Record result
            problem_result = {
                'problem_id': i,
                'question': question,
                'gold_answer': gold_answer,
                'agent_answer': result['answer'],
                'success': result['success'],
                'num_steps': len(result['trace'].steps)
            }

            # Save trace
            trace_file = os.path.join(output_dir, f"trace_{i}.json")
            result['trace'].to_json(trace_file)
            problem_result['trace_file'] = trace_file

            # Analyze failures with CausalFlow
            if not result['success'] and analyze_failures:
                print("\n  Running CausalFlow analysis...")
                try:
                    analysis = self.causal_flow.analyze_trace(
                        result['trace'],
                        skip_repair=False,
                        skip_critique=False
                    )

                    # Generate report
                    report_file = os.path.join(output_dir, f"analysis_{i}.txt")
                    self.causal_flow.generate_full_report(report_file)

                    # Export results
                    results_file = os.path.join(output_dir, f"analysis_{i}.json")
                    self.causal_flow.export_results(results_file)

                    problem_result['causal_analysis'] = {
                        'report_file': report_file,
                        'results_file': results_file,
                        'causal_steps': list(analysis.get('attribution', {}).keys()) if analysis else []
                    }

                    stats['analyzed'] += 1
                    print(f"  Analysis saved to {report_file}")

                except Exception as e:
                    print(f"  Error during analysis: {e}")
                    problem_result['causal_analysis'] = {'error': str(e)}

            # Update statistics
            if result['success']:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1

            stats['results'].append(problem_result)

        # Calculate metrics
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        stats['error_rate'] = stats['incorrect'] / stats['total'] if stats['total'] > 0 else 0
        stats['analysis_coverage'] = stats['analyzed'] / stats['incorrect'] if stats['incorrect'] > 0 else 0

        # Save summary
        summary_file = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total problems: {stats['total']}")
        print(f"Correct: {stats['correct']} ({stats['accuracy']*100:.1f}%)")
        print(f"Incorrect: {stats['incorrect']} ({stats['error_rate']*100:.1f}%)")
        print(f"Failures analyzed: {stats['analyzed']}/{stats['incorrect']}")
        print(f"\nResults saved to: {output_dir}/")
        print(f"Summary: {summary_file}")

        return stats

    def generate_benefits_report(self, output_dir: str = "gsm8k_results"):
        """
        Generate a report highlighting the benefits of CausalFlow.

        Args:
            output_dir: Directory containing experiment results
        """
        summary_file = os.path.join(output_dir, "experiment_summary.json")

        if not os.path.exists(summary_file):
            print("No summary file found. Run experiment first.")
            return

        with open(summary_file, 'r') as f:
            stats = json.load(f)

        report = []
        report.append("=" * 80)
        report.append("CausalFlow Benefits Report for GSM8K")
        report.append("=" * 80)
        report.append("")

        report.append("## Overview")
        report.append(f"- Total problems attempted: {stats['total']}")
        report.append(f"- Accuracy: {stats['accuracy']*100:.1f}%")
        report.append(f"- Failures: {stats['incorrect']}")
        report.append(f"- Failures analyzed by CausalFlow: {stats['analyzed']}")
        report.append("")

        report.append("## Benefits Demonstrated")
        report.append("")

        report.append("### 1. Automated Failure Diagnosis")
        report.append(f"   CausalFlow automatically identified the root causes of {stats['analyzed']} failures,")
        report.append("   showing which specific reasoning or calculation steps were causally responsible.")
        report.append("")

        report.append("### 2. Counterfactual Repairs")
        report.append("   For each failure, CausalFlow generated minimal edits that would have")
        report.append("   prevented the error, providing actionable insights for improvement.")
        report.append("")

        report.append("### 3. Multi-Agent Validation")
        report.append("   Causal claims were validated through multi-agent critique, ensuring")
        report.append("   robustness and reducing false positives in attribution.")
        report.append("")

        report.append("### 4. Detailed Trace Analysis")
        report.append("   Every step of the agent's reasoning was captured and analyzed,")
        report.append("   including:")
        report.append("   - Chain-of-thought reasoning steps")
        report.append("   - Calculator tool calls and responses")
        report.append("   - Dependencies between steps")
        report.append("   - Final answers and comparisons with gold answers")
        report.append("")

        report.append("## Example Failures Analyzed")
        report.append("")

        for i, result in enumerate(stats['results'][:5]):  # Show first 5
            if not result['success'] and 'causal_analysis' in result:
                report.append(f"### Problem {result['problem_id'] + 1}")
                report.append(f"Question: {result['question'][:100]}...")
                report.append(f"Expected: {result['gold_answer']}")
                report.append(f"Got: {result['agent_answer']}")

                if 'causal_steps' in result['causal_analysis']:
                    causal_steps = result['causal_analysis']['causal_steps']
                    report.append(f"Causal steps identified: {len(causal_steps)}")
                    report.append(f"See detailed analysis: {result['causal_analysis']['report_file']}")

                report.append("")

        report.append("=" * 80)

        # Save report
        report_file = os.path.join(output_dir, "causalflow_benefits.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"\nBenefits report saved to: {report_file}")
        print('\n'.join(report))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run GSM8K experiment with CausalFlow analysis"
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=5,
        help="Number of problems to solve (default: 5)"
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help="Use sample data instead of HuggingFace dataset"
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help="Skip CausalFlow analysis of failures"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='gsm8k_results',
        help="Output directory for results (default: gsm8k_results)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Check API key
    api_key = os.getenv("OPENROUTER_SECRET_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_SECRET_KEY not found in .env file")
        print("Please set your OpenRouter API key in .env")
        return

    # Run experiment
    experiment = GSM8KExperiment(api_key=api_key)

    stats = experiment.run_experiment(
        num_rows=args.num_rows,
        use_sample=args.use_sample,
        analyze_failures=not args.no_analysis,
        output_dir=args.output_dir
    )

    # Generate benefits report
    if not args.no_analysis and stats['analyzed'] > 0:
        experiment.generate_benefits_report(args.output_dir)


if __name__ == "__main__":
    main()
