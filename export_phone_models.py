"""Export the trained weights for the static phone demo (no retraining)."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vision import *
OUT=ROOT/'phone-site'/'models';OUT.mkdir(parents=True,exist_ok=True)

def export(name,layers,extra):
 arrays=[];offset=0;meta={};spec={}
 def pack(layer,key):
  nonlocal offset
  ws=[]
  for i,w in enumerate(layer.get_weights()):
   a=np.asarray(w,dtype='<f2');arrays.append(a.ravel());ws.append({'offset':offset,'size':a.size,'shape':list(a.shape)});offset+=a.size
  cfg=layer.get_config()
  spec[key]={'type':layer.__class__.__name__,'config':cfg,'weights':ws}
  return key
 extra=extra(pack)
 data=np.concatenate(arrays).astype('<f2').tobytes();shards=[]
 for i,start in enumerate(range(0,len(data),4*1024*1024)):
  filename=f'{name}-{i:02d}.bin';part=data[start:start+4*1024*1024];(OUT/filename).write_bytes(part);shards.append({'file':filename,'bytes':len(part),'sha256':hashlib.sha256(part).hexdigest()})
 manifest={'format':'device-lab-fp16-v1','architecture':name,'classes':CLASSES,'dtype':'float16','byteOrder':'little','elements':offset,'bytes':len(data),'shards':shards,'layers':spec,**extra}
 (OUT/f'{name}.json').write_text(json.dumps(manifest,separators=(',',':'),default=lambda x:np.asarray(x).tolist()),encoding='utf-8')
 print('EXPORTED',name,len(data),'bytes',flush=True)
 return manifest

def main():
 base=load_baseline();backbone=load_backbone();head=load_head('robust')
 mobile=next(l for l in base.layers if isinstance(l,tf.keras.Model) and len(l.layers)>10)
 def mobile_extra(pack):
  cfg=mobile.get_config();nodes=[]
  def inbound(value):
   if isinstance(value,dict):
    if value.get('class_name')=='__keras_tensor__':return [value['config']['keras_history'][0]]
    return sum((inbound(v) for v in value.values()),[])
   if isinstance(value,(list,tuple)):return sum((inbound(v) for v in value),[])
   return []
  for layer,config in zip(mobile.layers,cfg['layers']):
   pack(layer,layer.name);nodes.append({'name':layer.name,'inputs':inbound(config.get('inbound_nodes',[]))})
  return {'nodes':nodes,'output':mobile.layers[-1].name,'head':pack(base.layers[-1],'classifier')}
 export('mobilenet',[],mobile_extra)
 def conv_extra(pack):
  pre=backbone.get_layer('convnext_tiny_prestem_normalization').get_config()
  stem=backbone.get_layer('convnext_tiny_stem');stemkeys=[pack(l,'stem/'+l.name) for l in stem.layers]
  stages=[]
  for si,depth in enumerate([3,3,9,3]):
   down=[]
   if si:
    block=backbone.get_layer(f'convnext_tiny_downsampling_block_{si-1}');down=[pack(l,'down/'+l.name) for l in block.layers]
   blocks=[]
   for bi in range(depth):
    prefix=f'convnext_tiny_stage_{si}_block_{bi}_'
    blocks.append({suffix:pack(backbone.get_layer(prefix+suffix),prefix+suffix) for suffix in ['depthwise_conv','layernorm','pointwise_conv_1','pointwise_conv_2','layer_scale']})
   stages.append({'down':down,'blocks':blocks})
  return {'preprocessing':{'mean':pre['mean'],'variance':pre['variance']},'stem':stemkeys,'stages':stages,'final_norm':pack(backbone.layers[-1],'final_norm'),'head':pack(head.layers[-1],'classifier')}
 export('convnext',[],conv_extra)
 # Reference tensors allow parity checking independently of JPEG/canvas decoders.
 p=sorted((ROOT/'dataset/test/monitor').glob('*.jpg'))[0]
 with Image.open(p) as im:x=array(im)
 testdir=ROOT/'.cache'/'browser-validation';testdir.mkdir(parents=True,exist_ok=True)
 (testdir/'input.f32').write_bytes(x.astype('<f4').tobytes())
 fmap=backbone(x[None],training=False).numpy();prob=head(fmap.mean((1,2)),training=False).numpy()
 ref={'classes':CLASSES,'baseline':base(x[None],training=False).numpy()[0].tolist(),'modern':prob[0].tolist(),'map_shape':list(fmap.shape),'cam_class':int(prob.argmax())}
 (testdir/'map.f32').write_bytes(fmap.astype('<f4').tobytes())
 (testdir/'reference.json').write_text(json.dumps(ref,indent=2))
 print('Export and parity references ready.',flush=True)
if __name__=='__main__':main()
