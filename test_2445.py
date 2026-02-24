import urllib.request
import ssl
import certifi

url_ldraw = "https://cdn.rebrickable.com/media/parts/ldraw/14/2445.png"
url_elements = "https://cdn.rebrickable.com/media/parts/elements/2445.jpg"
url_bricklink = "https://img.bricklink.com/ItemImage/PN/0/2445.png"

context = ssl.create_default_context(cafile=certifi.where())

for url in [url_ldraw, url_elements, url_bricklink]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, context=context)
        print(f"✅ {url} - {res.status}")
    except Exception as e:
        print(f"❌ {url} - {e}")
