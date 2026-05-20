import urllib.request, json
url='http://127.0.0.1:8000/chat'
msgs=['I need help','I need help with calculus']
for msg in msgs:
    data=json.dumps({'session_id':'session_test_1','message':msg}).encode()
    req=urllib.request.Request(url,data,headers={'Content-Type':'application/json'})
    try:
        res=urllib.request.urlopen(req,timeout=10)
        print(msg+' -> '+ res.read().decode())
    except Exception as e:
        print('Error for',msg,':',e)
