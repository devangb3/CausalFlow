## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))
- (Optional) MongoDB for storing runs; Docker for MBPP/Humaneval reexecution.

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CF-Implementation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key and MongoDB URI
   ```

   Your `.env` file should contain:
   ```
   OPENROUTER_SECRET_KEY=your_api_key_here
   MONGODB_URI=mongodb://localhost:27017/causalflow
   ```

   For MongoDB Atlas (cloud):
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/causalflow?retryWrites=true&w=majority
   ```

### Quick Docs
- Experiment roundup and current results: `EXPERIMENTS_OVERVIEW.md`
- Sample end-to-end analysis output: `causalflow_report.txt` (generated via `CausalFlow.generate_full_report`)

## Quick Start

Run experiments (requires `.env` with `OPENROUTER_SECRET_KEY`; web tasks also need `SERPER_API_KEY`, MBPP/Humaneval need Docker):
- GSM8K: `python experiments/gsm8k/run_gsm8k_experiment.py`
- MBPP: `python experiments/mbpp/run_mbpp_experiment.py`
- Humaneval: `python experiments/humaneval/run_humaneval_experiment.py`
- BrowseComp: `python experiments/browsecomp/run_browsecomp_experiment.py`
- Seal QA Hard: `python experiments/browsecomp/run_sealqa_experiment.py`
- MedBrowseComp: `python experiments/browsecomp/run_medbrowsecomp_experiment.py`
