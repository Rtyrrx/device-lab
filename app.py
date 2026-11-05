"""Local-only demo: images stay in memory; no external prediction service."""
from vision import *
from flask import Flask,render_template,request,jsonify,send_from_directory
import io,base64,json,time,threading,platform
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=8*1024*1024
LOCK=threading.Lock(); bundle=None

def get_models():
 global bundle
 if bundle is None:
  baseline=load_baseline();backbone=load_backbone();head=load_head('robust')
  forward=tf.function(lambda x: backbone(x,training=False),input_signature=[tf.TensorSpec((None,224,224,3),tf.float32)])
  bundle=(baseline,backbone,head,forward)
  dummy=np.zeros((1,224,224,3),dtype=np.float32)
  bundle[0](dummy,training=False); bundle[2](tf.reduce_mean(bundle[3](dummy),axis=(1,2)),training=False)
 return bundle

def encoded(im):
 buff=io.BytesIO(); im.save(buff,format='JPEG',quality=90)
 return 'data:image/jpeg;base64,'+base64.b64encode(buff.getvalue()).decode()

@app.get('/')
def home(): return render_template('index.html')

@app.get('/api/results')
def results():
 p=ROOT/'results'/'metrics.json'
 return jsonify(json.loads(p.read_text())) if p.exists() else (jsonify(error='Experiments are still running.'),503)

@app.get('/sample/<name>')
def sample(name):
 if name not in CLASSES: return '',404
 p=sorted((ROOT/'dataset'/'test'/name).glob('*.jpg'))[0]
 return send_from_directory(p.parent,p.name)

@app.post('/api/predict')
def predict():
 try:
  name=request.form.get('sample'); cond=request.form.get('condition','clean')
  if cond not in CONDITIONS: return jsonify(error='Choose a listed image condition.'),400
  if name:
   if name not in CLASSES: return jsonify(error='Unknown sample.'),400
   p=sorted((ROOT/'dataset'/'test'/name).glob('*.jpg'))[0]
   with Image.open(p) as im: original=rgb(im)
  elif 'image' in request.files:
   with Image.open(request.files['image'].stream) as im:
    if im.width*im.height>25_000_000: return jsonify(error='Please use an image below 25 megapixels.'),400
    original=rgb(im)
  else: return jsonify(error='Choose an image first.'),400
  # Resize before demo corruption so displayed severity is consistent for uploads.
  image=condition(original.resize((300,300),Image.Resampling.BILINEAR),cond)
  x=array(image)[None]
  with LOCK:
   baseline,backbone,head,forward=get_models()
   start=time.perf_counter(); old=baseline(x,training=False).numpy()[0]; baseline_ms=(time.perf_counter()-start)*1000
   start=time.perf_counter(); fmap=forward(x).numpy()[0]; new=head(fmap.mean((0,1))[None],training=False).numpy()[0]; modern_ms=(time.perf_counter()-start)*1000
   pred=int(new.argmax()); overlay,evidence=cam_overlay(image,fmap,head,pred)
  def pack(prob,ms):
   index=int(prob.argmax())
   return {'label':CLASSES[index],'confidence':float(prob[index]),'ms':round(ms,1),'scores':[{'label':c,'probability':float(prob[i])} for i,c in enumerate(CLASSES)]}
  return jsonify(original=encoded(original.resize((448,448))),input=encoded(image.resize((448,448))),explanation=encoded(overlay),baseline=pack(old,baseline_ms),modern=pack(new,modern_ms),condition=cond,ground_truth=name,heatmap_available=bool(evidence.max()>0),environment={'device':'CPU','python':platform.python_version(),'tensorflow':tf.__version__})
 except (Image.UnidentifiedImageError,OSError,ValueError):
  return jsonify(error='This image could not be read. Try a JPG or PNG image.'),400

@app.errorhandler(413)
def too_large(error): return jsonify(error='This file is too large. Choose an image under 8 MB.'),413

if __name__=='__main__':
 print('Device Lab: http://127.0.0.1:5055',flush=True)
 # Bind only to this computer, without debug mode.
 app.run(host='127.0.0.1',port=5055,debug=False,threaded=True)
