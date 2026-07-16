import urllib.request, json
url='http://127.0.0.1:8000/chat'
msgs=[
    'How do I book?',
    'Is there a cost?',
    'How long are sessions?',
    'How can I cancel a booking?',
    'What about privacy of my data?',
    'Who do I contact for support?'
]
for msg in msgs:
    data=json.dumps({'session_id':'qa_test','message':msg}).encode()
    req=urllib.request.Request(url,data,headers={'Content-Type':'application/json'})
    try:
        res=urllib.request.urlopen(req,timeout=10)
        print(msg+' -> '+ res.read().decode())
    except Exception as e:
        print('Error for',msg,':',e)
