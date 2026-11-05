// Portable inference for the exported, already-trained Keras checkpoints.
// TensorFlow.js kernels; no training and no external prediction API.
let halfLookup;
export function float16ToFloat32(source) {
 const lookup=halfLookup||new Float32Array(65536);
 if(!halfLookup)for(let h=0;h<65536;h++) {const s=(h&32768)?-1:1,e=(h>>10)&31,f=h&1023;lookup[h]=e===0?s*Math.pow(2,-14)*(f/1024):e===31?(f?NaN:s*Infinity):s*Math.pow(2,e-15)*(1+f/1024);}
 halfLookup=lookup;const out=new Float32Array(source.length);for(let i=0;i<out.length;i++)out[i]=lookup[source[i]];return out;
}
export function createWeights(tf,manifest,half) {
 const weights={};
 for(const [name,layer] of Object.entries(manifest.layers))weights[name]=layer.weights.map(w=>tf.tensor(float16ToFloat32(half.subarray(w.offset,w.offset+w.size)),w.shape,'float32'));
 return weights;
}
export async function loadExport(tf,baseUrl,progress=()=>{}) {
 const res=await fetch(baseUrl);if(!res.ok)throw Error('Model manifest could not be downloaded.');const manifest=await res.json();
 const all=new Uint8Array(manifest.bytes);let offset=0;
 for(const shard of manifest.shards){const url=new URL(shard.file,new URL(baseUrl,location.href));const cache=globalThis.caches?await caches.open('device-lab-models-v1').catch(()=>null):null;const cacheKey=new URL(url);cacheKey.searchParams.set('sha256',shard.sha256);let r=cache?await cache.match(cacheKey):null;if(!r){r=await fetch(url);if(r.ok&&cache)await cache.put(cacheKey,r.clone()).catch(()=>{});}if(!r.ok)throw Error('A model download failed. Check your connection and retry.');const part=new Uint8Array(await r.arrayBuffer());if(part.byteLength!==shard.bytes)throw Error('Incomplete model download. Please retry.');if(globalThis.crypto?.subtle){const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',part))).map(b=>b.toString(16).padStart(2,'0')).join('');if(hash!==shard.sha256)throw Error('The model integrity check failed. Please reload.');}all.set(part,offset);offset+=part.byteLength;progress(offset,manifest.bytes);}
 return {manifest,weights:createWeights(tf,manifest,new Uint16Array(all.buffer))};
}
function dense(tf,x,w){const dims=x.shape;return tf.add(tf.matMul(x.reshape([-1,dims[dims.length-1]]),w[0]),w[1]).reshape([...dims.slice(0,-1),w[1].shape[0]]);}
function norm(tf,x,w,epsilon=1e-6){const mean=tf.mean(x,-1,true),variance=tf.mean(tf.square(tf.sub(x,mean)),-1,true);return tf.add(tf.mul(tf.mul(tf.sub(x,mean),tf.rsqrt(tf.add(variance,epsilon))),w[0]),w[1]);}
function layer(tf,x,key,model){const d=model.manifest.layers[key],w=model.weights[key],c=d.config;
 switch(d.type){
 case 'Conv2D':return tf.add(tf.conv2d(x,w[0],c.strides,c.padding),w[1]);
 case 'LayerNormalization':return norm(tf,x,w,c.epsilon);
 default:throw Error('Unsupported ConvNeXt layer: '+d.type);
 }
}
export async function runConvNext(tf,model,input,{includeFeatures=false}={}){
 const inferenceStarted=performance.now();const spec=model.manifest;let x=tf.tidy(()=>tf.div(tf.sub(input,tf.tensor1d(spec.preprocessing.mean)),tf.sqrt(tf.tensor1d(spec.preprocessing.variance))));
 try{
  for(const key of spec.stem){const next=tf.tidy(()=>layer(tf,x,key,model));x.dispose();x=next;}
  for(const stage of spec.stages){
   for(const key of stage.down){const next=tf.tidy(()=>layer(tf,x,key,model));x.dispose();x=next;}
   for(const block of stage.blocks){
    const next=tf.tidy(()=>{const w=model.weights;const dw=w[block.depthwise_conv];let y=tf.add(tf.depthwiseConv2d(x,dw[0].transpose([0,1,3,2]),1,'same'),dw[1]);y=norm(tf,y,w[block.layernorm],model.manifest.layers[block.layernorm].config.epsilon);y=dense(tf,y,w[block.pointwise_conv_1]);y=tf.mul(tf.mul(.5,y),tf.add(1,tf.erf(tf.div(y,Math.sqrt(2)))));y=dense(tf,y,w[block.pointwise_conv_2]);return tf.add(x,tf.mul(y,w[block.layer_scale][0]));});x.dispose();x=next;
   }
  }
  {const next=tf.tidy(()=>layer(tf,x,spec.final_norm,model));x.dispose();x=next;}
  const w=model.weights[spec.head];const probabilities=tf.tidy(()=>tf.softmax(dense(tf,tf.mean(x,[1,2]),w)));
  const scores=Array.from(await probabilities.data());probabilities.dispose();const predicted=scores.indexOf(Math.max(...scores));const inferenceMs=performance.now()-inferenceStarted;
  const heat=tf.tidy(()=>{const cw=w[0].slice([0,predicted],[768,1]).reshape([1,1,1,768]);const cam=tf.relu(tf.sum(tf.mul(x,cw),-1));return tf.div(cam,tf.maximum(tf.max(cam),1e-12));});
  const heatmap=Array.from(await heat.data());heat.dispose();const result={scores,heatmap,heatmapShape:[7,7],inferenceMs};if(includeFeatures)result.features=Array.from(await x.data());return result;
 }finally{x.dispose();}
}
export async function runMobileNet(tf,model,input){
 const inferenceStarted=performance.now();const {manifest,weights}=model;const values=new Map(),remaining=new Map();
 for(const node of manifest.nodes)for(const name of node.inputs)remaining.set(name,(remaining.get(name)||0)+1);
 let output;
 try{
  for(const node of manifest.nodes){const d=manifest.layers[node.name],w=weights[node.name],c=d.config;
   const out=tf.tidy(()=>{if(d.type==='InputLayer')return tf.sub(tf.div(input,127.5),1);const a=values.get(node.inputs[0]);switch(d.type){
    case 'Conv2D':{const b=tf.conv2d(a,w[0],c.strides,c.padding);return c.use_bias?tf.add(b,w[1]):b;}
    case 'DepthwiseConv2D':return tf.depthwiseConv2d(a,w[0],c.strides,c.padding);
    case 'BatchNormalization':return tf.batchNorm(a,w[2],w[3],w[1],w[0],c.epsilon);
    case 'ReLU':return tf.clipByValue(a,0,c.max_value===null?Infinity:c.max_value);
    case 'ZeroPadding2D':return tf.pad(a,[[0,0],c.padding[0],c.padding[1],[0,0]]);
    case 'Add':return tf.add(values.get(node.inputs[0]),values.get(node.inputs[1]));
    default:throw Error('Unsupported MobileNet layer: '+d.type);
   }});
   values.set(node.name,out);
   for(const key of node.inputs){remaining.set(key,remaining.get(key)-1);if(remaining.get(key)===0){values.get(key).dispose();values.delete(key);}}
  }
  output=values.get(manifest.output);const prob=tf.tidy(()=>tf.softmax(dense(tf,tf.mean(output,[1,2]),weights[manifest.head])));const scores=Array.from(await prob.data());prob.dispose();return {scores,inferenceMs:performance.now()-inferenceStarted};
 }finally{for(const value of values.values())value.dispose();}
}
export function disposeModel(model){for(const list of Object.values(model.weights))for(const w of list)w.dispose();}
