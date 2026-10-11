# CMPSC 448 Midterm Project

## Who Wrote It? Identifying LLMs from Their Responses

This project uses machine learning to identify which LLM generated a response.

### Dataset
300 responses from ChatGPT, Gemini, and Qwen, using 100 prompts across five categories.

### Models
- Convolutional Neural Network (CNN)
- Long Short-Term Memory Network (LSTM)

### Research Questions
- RQ1: Identify which LLM generated a response
- RQ2: Compare input-only, output-only, and combined classification
- RQ3: Cross-domain classification (extra credit)
- RQ4: Linguistic differences between LLMs (extra credit)

### Results
The best model was the CNN using combined input and output, achieving 71.11% accuracy.

### How to Run
Install the machine learning dependencies:

`python -m pip install torch numpy pandas scikit-learn matplotlib`

Run the main experiments:

`python src/train_models.py`

Run the extra-credit analysis:

`python src/extra_credit_analysis.py`

### Project Report
The PDF report is located at `report/CMPSC448 Midterm Report.pdf`.