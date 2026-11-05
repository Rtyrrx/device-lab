"""Exercise real prediction routes without requiring a browser/network connection."""
from app import app,ROOT
from pathlib import Path
import json,io,base64
from PIL import Image
out={}
with app.test_client() as c:
 assert c.get('/').status_code==200
 assert c.get('/sample/monitor').status_code==200
 assert c.get('/sample/invalid').status_code==404
 assert c.post('/api/predict',data={}).status_code==400
 assert c.post('/api/predict',data={'sample':'monitor','condition':'invalid'}).status_code==400
 assert c.post('/api/predict',data={'image':(io.BytesIO(b'not an image'),'bad.jpg')},content_type='multipart/form-data').status_code==400
 for name in ['monitor','telephone','laptop','keyboard']:
  r=c.post('/api/predict',data={'sample':name,'condition':'clean'})
  assert r.status_code==200,r.get_data(as_text=True)
  d=r.get_json();assert abs(sum(x['probability'] for x in d['modern']['scores'])-1)<1e-5
  for key in ['input','explanation']:
   data=base64.b64decode(d[key].split(',')[1]); im=Image.open(io.BytesIO(data));assert im.size==(448,448)
  out[name]={'old':d['baseline']['label'],'new':d['modern']['label'],'new_confidence':d['modern']['confidence']}
 p=next((ROOT/'dataset/test/keyboard').glob('*.jpg'))
 r=c.post('/api/predict',data={'image':(io.BytesIO(p.read_bytes()),'keyboard.jpg'),'condition':'blur'},content_type='multipart/form-data');assert r.status_code==200
 out['uploaded_blurred_image']={'status':r.status_code,'prediction':r.get_json()['modern']['label']}
(ROOT/'.cache').mkdir(exist_ok=True)
(ROOT/'.cache'/'app_smoke_check.json').write_text(json.dumps(out,indent=2))
print('PASS: app page, sample routes, invalid inputs, four real predictions, heatmaps, uploaded image and stress mode.')
print(json.dumps(out,indent=2))
