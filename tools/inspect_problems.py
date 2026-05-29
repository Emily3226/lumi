import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.contest_retriever import get_by_contest_year

problems = get_by_contest_year('Euclid', 2016, n=5)
print('count:', len(problems))
if problems:
    for k in ['pdf_path','problem_number','page_number','solution_pdf_path']:
        print(k, '->', problems[0].get(k))
    print('\nSample problems:')
    for p in problems[:3]:
        print(p['problem_number'], p['pdf_path'])
