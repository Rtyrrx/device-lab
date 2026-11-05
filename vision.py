"""Shared models and image handling. Pixel inputs are RGB float32 in [0,255]."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
os.environ.setdefault('KERAS_HOME',str(ROOT/'.cache'/'keras'))
import numpy as np
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import tensorflow as tf
try:
 tf.config.threading.set_intra_op_parallelism_threads(4)
 tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError: pass
CLASSES=['monitor','telephone','laptop','keyboard']
CONDITIONS=['clean','dim','blur','noise','occlusion']

def rgb(image):
 return ImageOps.exif_transpose(image).convert('RGB')

def array(image):
 # Match Keras image_dataset_from_directory: direct bilinear resize.
 return np.asarray(rgb(image).resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)

def condition(image,name,seed=42):
 image=rgb(image)
 if name=='clean': return image.copy()
 if name=='dim': return ImageEnhance.Brightness(image).enhance(0.35)
 if name=='blur': return image.filter(ImageFilter.GaussianBlur(3.0))
 if name=='noise':
  a=np.asarray(image,dtype=float); rng=np.random.default_rng(seed)
  return Image.fromarray(np.uint8(np.clip(a+rng.normal(0,25,a.shape),0,255)))
 if name=='occlusion':
  a=np.asarray(image).copy(); h,w=a.shape[:2]; a[int(.35*h):int(.65*h),int(.35*w):int(.65*w)]=128
  return Image.fromarray(a)
 raise ValueError('Unknown condition')

def training_variant(image,rng):
 image=rgb(image)
 if rng.random()<.5: image=ImageOps.mirror(image)
 image=ImageEnhance.Brightness(image).enhance(float(rng.uniform(.45,1.3)))
 image=ImageEnhance.Contrast(image).enhance(float(rng.uniform(.75,1.25)))
 if rng.random()<.5: image=image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.3,2.3))))
 a=np.asarray(image,dtype=float)
 if rng.random()<.5: a+=rng.normal(0,float(rng.uniform(3,18)),a.shape)
 return Image.fromarray(np.uint8(np.clip(a,0,255)))

def load_baseline():
 from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
 return tf.keras.models.load_model(ROOT/'models'/'a1_original.keras',custom_objects={'preprocess_input':preprocess_input},compile=False)

def load_backbone(pretrained=False):
 model=tf.keras.applications.ConvNeXtTiny(include_top=False,include_preprocessing=True,weights=str(ROOT/'models'/'convnext_imagenet_notop.h5') if pretrained and (ROOT/'models'/'convnext_imagenet_notop.h5').exists() else ('imagenet' if pretrained else None),input_shape=(224,224,3),pooling=None)
 model.trainable=False
 if not pretrained: model.load_weights(ROOT/'models'/'convnext_backbone.weights.h5')
 return model

def new_head():
 return tf.keras.Sequential([tf.keras.layers.Input((768,)),tf.keras.layers.Dense(4,activation='softmax',name='device_classifier')])

def load_head(variant='robust'):
 head=new_head(); head.load_weights(ROOT/'models'/f'{variant}_head.weights.h5'); return head

def cam_overlay(image,feature_map,head,class_index):
 weights=head.layers[-1].get_weights()[0][:,class_index]
 raw=np.einsum('hwc,c->hw',feature_map,weights)
 evidence=np.maximum(raw,0); maximum=float(evidence.max())
 if maximum>0: evidence/=maximum
 from matplotlib import colormaps
 heat=np.uint8(colormaps['inferno'](evidence)[...,:3]*255)
 original=rgb(image).resize((448,448))
 heat=Image.fromarray(heat).resize(original.size,Image.Resampling.BILINEAR)
 return Image.blend(original,heat,.42),evidence
