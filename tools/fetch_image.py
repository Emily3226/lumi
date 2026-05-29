import requests, base64, sys
url = 'http://127.0.0.1:8000/contest/page-image'
params = {
    'pdf_path': r'C:/Users/ezhan/lumi/contests/Euclid/2016EuclidContest.pdf',
    'prob_num': 1,
    'contest': 'Euclid',
    'scale': 2.0,
}
try:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    img_b = base64.b64decode(j['image_base64'])
    with open('out_problem.png','wb') as f:
        f.write(img_b)
    print('Wrote out_problem.png, bytes=', len(img_b), 'cropped=', j.get('cropped'), 'page=', j.get('page'))
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
