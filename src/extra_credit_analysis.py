"""CMPSC 448 optional RQ3 and RQ4 analyses.
Run from the project root: python src/extra_credit_analysis.py
Requires existing src/train_models.py and data/llm_responses.csv.
Writes new outputs only to results/extra_credit/.
"""
from pathlib import Path
import json
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import train_models as tm

DEST = ROOT / 'results' / 'extra_credit'
DEST.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data' / 'llm_responses.csv'
df = pd.read_csv(DATA)
if df[['prompt_id','category','LLM_name','LLM_input','LLM_output']].isna().any().any():
    raise ValueError('Missing required data cells')

# RQ4: simple, reproducible descriptive characteristics.
def metrics(text):
    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    lines = text.splitlines()
    bullets = sum(bool(re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', s)) for s in lines)
    headings = sum(bool(re.match(r'^\s*#{1,6}\s+', s)) for s in lines)
    return pd.Series({
        'word_count': len(words),
        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        'mean_words_per_sentence': len(words) / len(sentences) if sentences else 0,
        'bullet_or_numbered_lines': bullets,
        'markdown_heading_lines': headings,
    })

stats = pd.concat([df[['prompt_id','category','LLM_name']].reset_index(drop=True),
                   df.LLM_output.apply(metrics).reset_index(drop=True)], axis=1)
stats.to_csv(DEST / 'rq4_response_features.csv', index=False)
summary = stats.groupby('LLM_name').agg(
    samples=('word_count', 'size'),
    mean_word_count=('word_count','mean'),
    median_word_count=('word_count','median'),
    mean_unique_word_ratio=('unique_word_ratio','mean'),
    mean_sentence_words=('mean_words_per_sentence','mean'),
    mean_bullet_lines=('bullet_or_numbered_lines','mean'),
    pct_responses_with_bullets=('bullet_or_numbered_lines', lambda s: 100*(s.gt(0).mean())),
    pct_responses_with_markdown_headings=('markdown_heading_lines', lambda s: 100*(s.gt(0).mean())),
).round(3)
summary.to_csv(DEST / 'rq4_model_summary.csv')
for column, title, fname in [
    ('mean_word_count','Average words per response','rq4_word_counts.png'),
    ('mean_unique_word_ratio','Average unique-word ratio','rq4_lexical_diversity.png'),
    ('pct_responses_with_bullets','Responses using bullet or numbered lists (%)','rq4_lists.png'),
]:
    fig, ax = plt.subplots(figsize=(7,4))
    summary[column].plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Model family')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(DEST / fname, dpi=160)
    plt.close(fig)
print('\nRQ4 model summary:\n', summary.to_string(), flush=True)

# RQ3: hold out all 20 creative-writing prompts; train on other categories.
# This is a DIFFERENT experiment from the existing random prompt-held-out split.
# The models are trained from scratch with NO access to writing responses.
writing = df['category'].astype(str).str.lower().str.contains('creative|writing')
if writing.sum() != 60:
    raise ValueError(f'Expected 60 creative-writing responses (20 prompts x 3 models), found {writing.sum()}. Check category names.')
train_pool = df.loc[~writing].copy()
test = df.loc[writing].copy()
ids = sorted(train_pool.prompt_id.unique())
if len(ids) < 4:
    raise ValueError('Not enough non-writing prompts for training/validation')
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(ids, test_size=0.20, random_state=tm.SEED)
train = train_pool[train_pool.prompt_id.isin(train_ids)].copy()
val = train_pool[train_pool.prompt_id.isin(val_ids)].copy()
vocab = tm.build_vocab(train)
print(f'\nRQ3: train {len(train)} / val {len(val)} / unseen writing test {len(test)}', flush=True)

outputs=[]
for name, model_class in [('CNN', tm.CNN), ('LSTM', tm.LSTM)]:
    mode='output'
    torch.manual_seed(tm.SEED)
    model=model_class(len(vocab)).to(tm.DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=tm.LR)
    criterion=torch.nn.CrossEntropyLoss()
    loaders={
        'train': DataLoader(tm.ResponsesDataset(train,vocab,mode), batch_size=tm.BATCH_SIZE,shuffle=True),
        'val': DataLoader(tm.ResponsesDataset(val,vocab,mode), batch_size=tm.BATCH_SIZE),
        'test': DataLoader(tm.ResponsesDataset(test,vocab,mode), batch_size=tm.BATCH_SIZE),
    }
    best=-1
    best_weights=None
    best_epoch=None
    from copy import deepcopy
    for epoch in range(1, tm.EPOCHS+1):
        model.train()
        for x,y in loaders['train']:
            x,y=x.to(tm.DEVICE),y.to(tm.DEVICE)
            optimizer.zero_grad()
            loss=criterion(model(x),y)
            loss.backward()
            optimizer.step()
        score=tm.evaluate(model,loaders['val'])
        if score['macro_f1'] > best:
            best=score['macro_f1'];best_epoch=epoch
            best_weights=deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    score=tm.evaluate(model,loaders['test'])
    outputs.append({'model':name,'train_domain':'non-creative-writing',
                    'test_domain':'creative-writing','train_rows':len(train),
                    'validation_rows':len(val),'test_rows':len(test),
                    'best_epoch':best_epoch,'validation_macro_f1':best,
                    'unseen_domain_accuracy':score['accuracy'],
                    'unseen_domain_macro_f1':score['macro_f1']})
    pd.DataFrame({'prompt_id':test.prompt_id.to_numpy(),
                  'actual':[tm.LABELS[i] for i in score['truth']],
                  'predicted':[tm.LABELS[i] for i in score['predictions']]}).to_csv(
                  DEST / f'rq3_{name}_predictions.csv', index=False)
    print(f'RQ3 {name}: accuracy={score["accuracy"]:.3f}, macro_f1={score["macro_f1"]:.3f}', flush=True)

pd.DataFrame(outputs).to_csv(DEST/'rq3_cross_domain_results.csv', index=False)
(DEST/'rq3_experiment_description.txt').write_text(
    'Train: non-creative-writing categories; 80% prompt IDs for training and 20% for validation. '
    'Test: all creative-writing prompt IDs, unseen in model training. '
    'Input: LLM_output only. CNN and LSTM initialized from scratch. '
    'Validation macro F1 selects best epoch; held-out domain is tested once. '
    'This is an exploratory domain shift experiment with only one held-out domain.\n'
)
print('\nDONE: Extra credit outputs saved to', DEST)
