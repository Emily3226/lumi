import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.contest_retriever import get_by_contest_year
import requests, base64
problems = get_by_contest_year('Euclid', 2016, n=1)
if not problems:
    print('no problems')
    raise SystemExit(1)
prob = problems[0]
params = {
    'pdf_path': prob['pdf_path'],
    'prob_num': prob['problem_number'] or 1,
    'contest': prob['contest'],
    'scale': 2.0,
}
print('requesting', params)
r = requests.get('http://127.0.0.1:8000/contest/page-image', params=params, timeout=30)
r.raise_for_status()
j = r.json()
img = base64.b64decode(j['image_base64'])
open('out_problem.png','wb').write(img)
print('Wrote out_problem.png bytes=', len(img))
