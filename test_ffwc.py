import requests

url = 'https://www.ffwc.gov.bd/index.php/bdlevel/api/station/SW116'

try:
    r = requests.get(url, timeout=10)
    print('Status Code:', r.status_code)
    print('Response (first 500 chars):')
    print(r.text[:500])
except Exception as e:
    print('Error:', e)