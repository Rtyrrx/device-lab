// Keep CPU kernels and weight decoding off the UI thread.
importScripts('./tf.min.js','./tf-backend-wasm.js');
let engine,baseline,modern,busy=false;
const tell=(type,data={})=>postMessage({type,...data});
async function ready(){
 if(!engine)engine=(async()=>{tf.wasm.setWasmPaths(new URL('./',self.location.href).href);tf.wasm.setThreadsCount(1);await tf.setBackend('wasm');await tf.ready();if(tf.getBackend()!=='wasm')throw Error('WebAssembly could not start. Please update your browser.');return import('./inference.mjs?v=6');})().catch(e=>{engine=null;throw e;});
 return engine;
}
onmessage=async({data})=>{
 if(busy)return;busy=true;let input;
 try{
  tell('stage',{text:'Starting vision engine…'});const api=await ready();
  const progress=name=>(n,total)=>tell('progress',{text:`Loading ${name} · ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`});
  if(!baseline)baseline=await api.loadExport(tf,new URL('../models/mobilenet.json',self.location.href),progress('MobileNet'));
  tell('stage',{text:'Analyzing with MobileNet…'});
  input=tf.tidy(()=>tf.image.resizeBilinear(tf.tensor3d(new Uint8Array(data.pixels),[300,300,4],'int32').slice([0,0,0],[300,300,3]),[224,224],false,true).toFloat().expandDims(0));
  tell('baseline',{result:await api.runMobileNet(tf,baseline,input)});
  if(!modern)modern=await api.loadExport(tf,new URL('../models/convnext.json',self.location.href),progress('ConvNeXt'));
  tell('stage',{text:'Analyzing with ConvNeXt…'});
  const result=await api.runConvNext(tf,modern,input);
  const simd=await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT');
  tell('complete',{result,runtime:`CPU / WebAssembly${simd?' · SIMD':' · SIMD unavailable'}`});
 }catch(e){tell('failure',{text:e.message||'Analysis failed. Please retry.'});}
 finally{input?.dispose();busy=false;}
};
