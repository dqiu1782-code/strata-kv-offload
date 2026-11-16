"""Quality test of the block selection inside strata-q36 (STRATA_Q36_SEL=1) against its own dense attention.

For each question over a long context made of public source files:
  dense   greedy generation (reference A)
  sel     free-running greedy generation: tokens equal to A / first difference
  sel-tf  teacher-forced on A: share of steps whose top-1 equals A
Needle runs: a passcode inserted at a depth; does the answer contain it (dense and sel)?

Set STRATA_Q36_KVHOST=1 and STRATA_Q36_KVPOOL=256 in the environment to test the offloaded K / V path.
All paths can be overridden with environment variables (see below).
"""
import glob
import json
import os
import subprocess
import sys
import time

from transformers import AutoTokenizer

EXE = os.environ.get('SEL_EXE', r'E:\llm\strata_build\strata-q36.exe')
MODEL = os.environ.get('SEL_MODEL', r'E:\llm\models\q36gguf\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf')
TOKDIR = os.environ.get('SEL_TOKDIR', r'E:\llm\models\Qwen3.6-35B-A3B-GPTQ-Int4')   # any Qwen3.6 tokenizer
CUDA = os.environ.get('SEL_CUDA', r'E:\llm\cuda\v13.0')
CTX = int(os.environ.get('SEL_CTX', '60000'))
STEPS = int(os.environ.get('SEL_STEPS', '160'))
OUT = os.environ.get('SEL_OUT', r'E:\llm\kvhybrid\sel_eval_public_%d' % CTX)
# default corpus: the Strata source tree itself (public, MIT), without src/qwen35 (the files this project changes)
SRC = os.environ.get('SEL_SRC', r'E:\llm\strata\include;E:\llm\strata\src').split(';')
SKIP = [s.lower() for s in os.environ.get('SEL_SKIP', r'E:\llm\strata\src\qwen35').split(';') if s]
EXTS = ('.h', '.hpp', '.cuh', '.cu', '.cpp', '.c', '.py', '.md')

QS = {
    'q1': 'Explain the overall structure of this code base: what each top-level folder does and how the parts connect. Mention specific file names.',
    'q2': 'Which programming languages are used in this code base, and what is each one used for? Mention specific file names.',
    'q3': 'Which parts of the code manage the KV cache and memory on the GPU and the host? Describe what they do and mention specific file names.',
    'q4': 'How would I build and run this code base? List the steps and mention the specific files and commands involved.',
}
CODE = 'MAGENTA-4172'
NEEDLE_Q = 'What is the secret deployment passcode mentioned in the project files? Reply with only the passcode.'


def build_context(tok, srcs, ctx):
    files = []
    for s in srcs:
        for p in sorted(glob.glob(os.path.join(s, '**', '*'), recursive=True)):
            if any(p.lower().startswith(k) for k in SKIP):
                continue
            if p.endswith(EXTS) and os.path.isfile(p) and os.path.getsize(p) < 200_000:
                files.append(p)
    parts, used = [], 0
    root = os.path.dirname(srcs[0].rstrip('\\/'))
    for p in files:
        try:
            txt = open(p, encoding='utf-8').read()
        except Exception:
            continue
        piece = f'\n\n===== FILE: {os.path.relpath(p, root)} =====\n{txt}'
        n = len(tok(piece, add_special_tokens=False).input_ids)
        if used + n > ctx:
            break
        parts.append(piece)
        used += n
    return 'Below are the source files of a project.' + ''.join(parts), used


os.makedirs(OUT, exist_ok=True)
env0 = dict(os.environ)
env0['PATH'] = CUDA + r'\bin;' + CUDA + r'\bin\x64;' + env0['PATH']
tok = AutoTokenizer.from_pretrained(TOKDIR)
body, body_tokens = build_context(tok, SRC, CTX)
print(json.dumps(dict(corpus=SRC, skip=SKIP, context_tokens=body_tokens,
                      kvhost=os.environ.get('STRATA_Q36_KVHOST'), kvpool=os.environ.get('STRATA_Q36_KVPOOL'))), flush=True)


def prompt_ids(question, depth=None):
    b = body
    if depth is not None:
        pos = int(len(b) * depth)
        j = b.find('\n\n===== FILE:', pos)
        j = pos if j < 0 else j
        b = b[:j] + f'\n\n===== NOTE =====\nThe secret deployment passcode is {CODE}. Please remember it.' + b[j:]
    msgs = [{'role': 'user', 'content': b + '\n\n' + question}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=False).input_ids


def write_ids(path, ids):
    with open(path, 'w') as f:
        f.write(','.join(map(str, ids)))


def run(tag, pfile, n, sel, force=None):
    env = dict(env0)
    if sel:
        env['STRATA_Q36_SEL'] = '1'
    else:   # dense reference: K / V on the GPU, no selection
        for k in ('STRATA_Q36_SEL', 'STRATA_Q36_KVHOST', 'STRATA_Q36_KVPOOL'):
            env.pop(k, None)
    cmd = [EXE, '--model', MODEL, '--ctx', str(CTX + STEPS + 512), '--tokens', '@' + pfile, '--n', str(n)]
    if force:
        cmd += ['--force', '@' + force]
    t = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    open(os.path.join(OUT, tag + '.err'), 'w', encoding='utf-8').write(r.stderr)
    for line in r.stdout.splitlines():
        if line.startswith('gen:'):
            return [int(x) for x in line[4:].split()], time.time() - t
    raise RuntimeError('no gen line for ' + tag + ': ' + r.stderr[-500:])


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


for name, q in QS.items():
    ids = prompt_ids(q)
    pf = os.path.join(OUT, name + '.ids')
    write_ids(pf, ids)
    A, ta = run(name + '_dense', pf, STEPS, False)
    write_ids(os.path.join(OUT, name + '_A.ids'), A[:STEPS - 1])
    B, tb = run(name + '_sel', pf, STEPS, True)
    C, tc = run(name + '_seltf', pf, STEPS, True, os.path.join(OUT, name + '_A.ids'))
    same = sum(x == y for x, y in zip(A, B))
    tf = sum(x == y for x, y in zip(A, C)) / len(A)
    row = dict(q=name, prompt=len(ids), n=len(A), free_equal=same, first_diff=first_diff(A, B), tf_agree=round(tf, 4))
    print(json.dumps(row), flush=True)
    open(os.path.join(OUT, name + '_text.txt'), 'w', encoding='utf-8').write(
        '--- dense ---\n' + tok.decode(A) + '\n--- sel ---\n' + tok.decode(B) + '\n')

for depth in (0.2, 0.5, 0.8):
    ids = prompt_ids(NEEDLE_Q, depth)
    pf = os.path.join(OUT, 'needle.ids')
    write_ids(pf, ids)
    res = {}
    for sel in (False, True):
        g, _ = run('needle_%s_%s' % (depth, 'sel' if sel else 'dense'), pf, 24, sel)
        res['sel' if sel else 'dense'] = CODE in tok.decode(g)
    print(json.dumps(dict(needle_depth=depth, **res)), flush=True)
print('SEL_EVAL_DONE', flush=True)
