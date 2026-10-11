"""CMPSC 448 response collector: ChatGPT, Gemini, and Llama (OpenRouter).

Test one model: python collect_llm_responses_llama.py --provider Llama --limit 1
Test all three: python collect_llm_responses_llama.py --limit 3
Continue:       python collect_llm_responses_llama.py

Requires .env with OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY.
The script only requires keys for providers it actually calls. Saves after every answer.
"""
import argparse
import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
SOURCE = ROOT / "data" / "CMPSC 448 Midterm.xlsx"
OUTPUT = ROOT / 'CMPSC448_Responses.xlsx'
MODELS = {
    'ChatGPT': 'gpt-5-mini',
    'Gemini': 'gemini-3.5-flash-lite',
    'Qwen': 'qwen/qwen3.8-27b',
}
KEYS = {
    'ChatGPT': 'OPENAI_API_KEY',
    'Gemini': 'GEMINI_API_KEY',
    'Qwen': 'GROQ_API_KEY',
}

def client_for(name):
    key = os.getenv(KEYS[name])
    if not key:
        raise RuntimeError(f'Missing {KEYS[name]} in .env (needed for {name})')
    if name == 'Qwen':
        return OpenAI(
            api_key=key,
            base_url='https://api.groq.com/openai/v1',
            timeout=60
        )
    if name == 'ChatGPT':
        return OpenAI(api_key=key, timeout=60)
    return genai.Client(api_key=key)

def request(name, prompt, client):
    if name == 'ChatGPT':
        for attempt in range(3):
            r = client.responses.create(
                model=MODELS[name],
                input=prompt,
                max_output_tokens=3000,
                reasoning={"effort": "low"}
            )

            answer = (r.output_text or "").strip()

            if answer:
                return answer

            print(
                f"ChatGPT empty response "
                f"(attempt {attempt + 1}/3). "
                f"Status: {r.status}, "
                f"Reason: {r.incomplete_details}"
            )

            if attempt < 2:
                time.sleep(5)

        raise RuntimeError(
            "ChatGPT returned no text after 3 attempts."
        )
    if name == 'Qwen':
        r = client.chat.completions.create(
            model=MODELS[name], messages=[{'role': 'user', 'content': prompt}],
            max_tokens=1000, temperature=0.7,
        )
        return (r.choices[0].message.content or '') if r.choices else ''
    r = client.models.generate_content(
        model=MODELS[name], contents=prompt,
        config={'max_output_tokens': 1000},
    )
    return r.text or ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=list(MODELS), help='Only collect for one model')
    parser.add_argument('--limit', type=int, help='Maximum NEW API calls in this run')
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f'Put this spreadsheet beside the script: {SOURCE.name}')
    df = pd.read_excel(OUTPUT if OUTPUT.exists() else SOURCE)
    df['LLM_output'] = df['LLM_output'].astype('object')
    required = ['prompt_id','category','LLM_name','LLM_input','LLM_output']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Spreadsheet missing columns: {missing_cols}')
    df['LLM_name'] = df['LLM_name'].replace({
    'Claude': 'Qwen',
    'Llama': 'Qwen'
    })
    if 'model_version' not in df.columns:
        df['model_version'] = ''
    if 'collection_status' not in df.columns:
        df['collection_status'] = ''
    if 'collection_date_utc' not in df.columns:
        df['collection_date_utc'] = ''
    clients = {}
    completed = 0
    for i, row in df.iterrows():
        name = str(row['LLM_name']).strip()
        if name not in MODELS:
            raise ValueError(f'Unexpected AI name on spreadsheet row {i+2}: {name}')
        if args.provider and name != args.provider:
            continue
        existing = row['LLM_output']
        if pd.notna(existing) and str(existing).strip():
            continue
        if args.limit is not None and completed >= args.limit:
            break
        prompt = str(row['LLM_input']).strip()
        if not prompt or prompt.lower() == 'nan':
            raise ValueError(f'Missing prompt on spreadsheet row {i+2}')
        if name not in clients:
            clients[name] = client_for(name)
        print(f'Row {i+2}: prompt {row["prompt_id"]} -> {name}', flush=True)
        try:
            answer = request(name, prompt, clients[name]).strip()
            if not answer:
                raise RuntimeError('Empty model response; check free model availability or output settings')
            df.at[i, 'LLM_output'] = answer
            df.at[i, 'model_version'] = MODELS[name]
            df.at[i, 'collection_status'] = 'complete'
            df.at[i, 'collection_date_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
            df.to_excel(OUTPUT, index=False)
            completed += 1
            print(f'Saved {len(answer)} characters.', flush=True)
        except Exception as e:
            print(f'Error: {type(e).__name__}: {e}', flush=True)
            print('Completed responses are saved. Check the issue and rerun.', flush=True)
            raise SystemExit(1)
        time.sleep(1.5)
    done = df['LLM_output'].notna() & df['LLM_output'].astype(str).str.strip().ne('')
    print(f'Newly collected: {completed}. Total: {done.sum()}/{len(df)}')
    print(f'Saved workbook: {OUTPUT.name}')
    if done.all():
        df.to_csv(ROOT / 'llm_responses.csv', index=False, encoding='utf-8')
        print('Exported llm_responses.csv')

if __name__ == '__main__':
    main()
