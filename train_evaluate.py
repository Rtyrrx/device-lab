"""Reproduce frozen-feature training, augmentation ablation and evaluation."""
from vision import *
import json,time,platform,csv,hashlib
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
CACHE=ROOT/'.cache'; CACHE.mkdir(exist_ok=True)
SEED=42
FORWARD=None
np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

def files(split):
 return [(p,i) for i,c in enumerate(CLASSES) for p in sorted((ROOT/'dataset'/split/c).glob('*.jpg'))]

def extract(backbone,items,key,augment=False):
 path=CACHE/f'{key}.npz'
 if path.exists():
  d=np.load(path); return d['x'],d['y']
 xs=[]; ys=[]; rng=np.random.default_rng(SEED)
 for start in range(0,len(items),8):
  batch=items[start:start+8]; ims=[]
  for p,label in batch:
   with Image.open(p) as im: ims.append(array(training_variant(im,rng) if augment else im))
   ys.append(label)
  maps=FORWARD(np.stack(ims)).numpy()
  xs.extend(maps.mean(axis=(1,2)))
  if start%80==0: print(f'Extract {key}: {start}/{len(items)}',flush=True)
 x=np.asarray(xs); y=np.asarray(ys); np.savez_compressed(path,x=x,y=y)
 return x,y

def fit_head(name,x,y,vx,vy):
 tf.keras.utils.set_random_seed(SEED)
 head=new_head(); head.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 rng=np.random.default_rng(SEED); records=[]; best=float('inf'); wait=0
 for epoch in range(80):
  order=rng.permutation(len(x)); head.reset_metrics()
  for start in range(0,len(x),32): head.train_on_batch(x[order[start:start+32]],y[order[start:start+32]])
  tp=head(x,training=False).numpy(); vp=head(vx,training=False).numpy()
  def ce(labels,p): return float(-np.log(np.clip(p[np.arange(len(labels)),labels],1e-7,1)).mean())
  row={'epoch':epoch+1,'loss':ce(y,tp),'accuracy':float(accuracy_score(y,tp.argmax(1))),'val_loss':ce(vy,vp),'val_accuracy':float(accuracy_score(vy,vp.argmax(1)))}
  records.append(row)
  if row['val_loss']<best-1e-5:
   best=row['val_loss']; wait=0; head.save_weights(ROOT/'models'/f'{name}_head.weights.h5')
  else: wait+=1
  if epoch%10==0: print(name,row,flush=True)
  if wait>=10: break
 head.load_weights(ROOT/'models'/f'{name}_head.weights.h5')
 (RESULTS/f'{name}_history.json').write_text(json.dumps(records,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(9,3.2))
 for ax,keys,title in zip(axes,[['accuracy','val_accuracy'],['loss','val_loss']],['Accuracy','Cross-entropy loss']):
  for k in keys: ax.plot([r['epoch'] for r in records],[r[k] for r in records],label=k)
  ax.set(xlabel='Epoch',title=title); ax.legend(); ax.grid(alpha=.2)
 fig.suptitle(f'ConvNeXt-Tiny: {name} head (backbone frozen)'); fig.tight_layout(); fig.savefig(RESULTS/f'{name}_curves.png',dpi=160); plt.close(fig)
 return head

def metrics(y,p):
 labels=p.argmax(1)
 return {'accuracy':float(accuracy_score(y,labels)),'macro_f1':float(f1_score(y,labels,average='macro',zero_division=0)),'correct':int((y==labels).sum()),'n':len(y),'confusion_matrix':confusion_matrix(y,labels,labels=range(4)).tolist(),'per_class':classification_report(y,labels,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0)}

def main():
 global FORWARD
 print('Loading ImageNet ConvNeXt-Tiny',flush=True)
 backbone=load_backbone(pretrained=not (ROOT/'models'/'convnext_backbone.weights.h5').exists())
 if not (ROOT/'models'/'convnext_backbone.weights.h5').exists(): backbone.save_weights(ROOT/'models'/'convnext_backbone.weights.h5')
 FORWARD=tf.function(lambda x: backbone(x,training=False),input_signature=[tf.TensorSpec((None,224,224,3),tf.float32)])
 tr=files('train'); va=files('validate'); te=files('test')
 tx,ty=extract(backbone,tr,'train_clean'); vx,vy=extract(backbone,va,'val_clean')
 # Two deterministic extra variants per TRAINING image only.
 ax,ay=extract(backbone,tr*2,'train_augmented_two',augment=True)
 # Visually reviewed dHash candidates: exclude a train/validation near-duplicate
 # monitor from TRAINING and a validation/test near-duplicate phone from VALIDATION.
 keep_train=np.array([not (p.parent.name=='monitor' and p.name=='01602417.jpg') for p,_ in tr])
 keep_val=np.array([not (p.parent.name=='telephone' and p.name=='025005029 - Copy - Copy.jpg') for p,_ in va])
 tx,ty=tx[keep_train],ty[keep_train]; vx,vy=vx[keep_val],vy[keep_val]
 keep_aug=np.tile(keep_train,2); ax,ay=ax[keep_aug],ay[keep_aug]
 plain=fit_head('plain',tx,ty,vx,vy)
 robust=fit_head('robust',np.concatenate([tx,ax]),np.concatenate([ty,ay]),vx,vy)
 baseline=load_baseline()
 all_results={'seed':SEED,'classes':CLASSES,'environment':{'python':platform.python_version(),'tensorflow':tf.__version__,'platform':platform.platform(),'processor':platform.processor(),'device':'CPU','intra_op_threads':4,'inter_op_threads':2},'protocol':{'test_images':60,'conditions':CONDITIONS,'training_originals':len(tx),'extra_training_variants':len(ax),'validation_images':len(vx),'duplicate_exclusions':'train/monitor/01602417.jpg; validate/telephone/025005029 - Copy - Copy.jpg','backbone':'ConvNeXt-Tiny ImageNet-1k, frozen','head':'768 -> 4 softmax','optimizer':'Adam 0.001','max_epochs':80,'early_stopping':'clean validation loss, patience 10; restore best','stress_warning':'Corruptions reuse the same 60 test images; they are not independent new samples.'},'models':{n:{} for n in ['a1_mobilenet','convnext_plain','convnext_robust']}}
 prediction_rows=[]
 for cond in CONDITIONS:
  print('Evaluate',cond,flush=True)
  images=[]; y=[]
  for j,(p,label) in enumerate(te):
   with Image.open(p) as im: images.append(array(condition(im,cond,SEED+j)))
   y.append(label)
  images=np.stack(images); y=np.asarray(y); maps=[]; bp=[]
  for start in range(0,len(images),8):
   b=images[start:start+8]; maps.extend(FORWARD(b).numpy().mean((1,2))); bp.extend(baseline(b,training=False).numpy())
  maps=np.asarray(maps)
  for name,p in [('a1_mobilenet',np.asarray(bp)),('convnext_plain',plain(maps,training=False).numpy()),('convnext_robust',robust(maps,training=False).numpy())]:
   all_results['models'][name][cond]=metrics(y,p)
   print(name,cond,all_results['models'][name][cond]['accuracy'],flush=True)
   for j,(path,label) in enumerate(te): prediction_rows.append({'model':name,'condition':cond,'file':path.relative_to(ROOT).as_posix(),'true':CLASSES[label],'predicted':CLASSES[int(p[j].argmax())],'confidence':float(p[j].max())})
  (RESULTS/'metrics.json').write_text(json.dumps(all_results,indent=2))
 # Warmed, batch-one timings exclude loading, file I/O and explanations.
 image=images[:1]; timers={}
 funcs={'a1_mobilenet':lambda:baseline(image,training=False).numpy(),'convnext_plain':lambda:plain(tf.reduce_mean(FORWARD(image),axis=(1,2)),training=False).numpy(),'convnext_robust':lambda:robust(tf.reduce_mean(FORWARD(image),axis=(1,2)),training=False).numpy()}
 for name,fn in funcs.items():
  for _ in range(3): fn()
  times=[]
  for _ in range(20):
   t=time.perf_counter(); fn(); times.append((time.perf_counter()-t)*1000)
  timers[name]={'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'std_ms':float(np.std(times)),'runs':20,'warmups':3}
 all_results['latency']=timers
 (RESULTS/'metrics.json').write_text(json.dumps(all_results,indent=2))
 with (RESULTS/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=prediction_rows[0]); w.writeheader(); w.writerows(prediction_rows)
 summary=[]
 for name,data in all_results['models'].items():
  for cond,m in data.items(): summary.append({'model':name,'condition':cond,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'correct':m['correct'],'n':m['n']})
 with (RESULTS/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,(name,data) in zip(axes,all_results['models'].items()):
  cm=np.array(data['clean']['confusion_matrix']); ax.imshow(cm,cmap='Blues',vmin=0,vmax=15)
  for i in range(4):
   for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>7 else 'black')
  ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,title=name,xlabel='Predicted',ylabel='True'); ax.tick_params(axis='x',rotation=35)
 fig.tight_layout(); fig.savefig(RESULTS/'confusion_matrices.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,3.8))
 for name,data in all_results['models'].items(): ax.plot(CONDITIONS,[100*data[c]['accuracy'] for c in CONDITIONS],marker='o',label=name)
 ax.set(ylabel='Accuracy (%)',ylim=(0,105),title='Same held-out images under controlled conditions'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(RESULTS/'robustness.png',dpi=160); plt.close(fig)
 # One example per class, with true CAM from the deployed robust model.
 fig,axes=plt.subplots(2,4,figsize=(10,5))
 examples=[]
 for i,c in enumerate(CLASSES):
  p=next(p for p,label in te if label==i)
  with Image.open(p) as im:
   im=rgb(im); fmap=FORWARD(array(im)[None]).numpy()[0]; prob=robust(fmap.mean((0,1))[None],training=False).numpy()[0]; pred=int(prob.argmax()); overlay,_=cam_overlay(im,fmap,robust,pred)
   axes[0,i].imshow(im); axes[1,i].imshow(overlay); axes[0,i].set_title('True: '+c); axes[1,i].set_title(f'{CLASSES[pred]} {prob[pred]:.1%}')
   examples.append({'file':p.relative_to(ROOT).as_posix(),'true':c,'predicted':CLASSES[pred],'confidence':float(prob[pred])})
  axes[0,i].axis('off'); axes[1,i].axis('off')
 fig.suptitle('Class Activation Maps: positive evidence for the predicted class'); fig.tight_layout(); fig.savefig(RESULTS/'explanations.png',dpi=160); plt.close(fig)
 (RESULTS/'explanation_examples.json').write_text(json.dumps(examples,indent=2))
 print('EXPERIMENTS COMPLETE',flush=True)
if __name__=='__main__': main()
