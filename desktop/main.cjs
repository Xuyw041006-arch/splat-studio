const { app, BrowserWindow, dialog, session, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const net = require('node:net');
const crypto = require('node:crypto');
const { startupOptions } = require('./startup.cjs');
const { backendRuntime } = require('./backend-runtime.cjs');
const { installSceneExport } = require('./scene-export.cjs');
const { authorizeProjectDirectory, ProjectAccessCancelled } = require('./project-access.cjs');
const {createLanguagePreferences,installLanguageIPC}=require('./language.cjs');
const preferences=createLanguagePreferences(path.join(app.getPath('userData'),'language.json'));
const t=text=>preferences.t(text);
let backend;
let stopping=false;
const token=crypto.randomBytes(32).toString('hex');
function freePort(){return new Promise((resolve,reject)=>{const server=net.createServer();server.once('error',reject);server.listen(0,'127.0.0.1',()=>{const port=server.address().port;server.close(()=>resolve(port));});});}
async function waitBackend(url){
  for(let i=0;i<180;i++){
    if(backend.exitCode!==null)throw new Error(t('本地引擎启动失败，请查看 backend.log。'));
    try{const res=await fetch(url+'/api/health',{headers:{'x-splat-token':token}});if(res.ok)return;}catch{}
    await new Promise(r=>setTimeout(r,500));
  }
  throw new Error(t('本地引擎启动超时，请查看 backend.log。'));
}
async function start(){
  const root=path.resolve(__dirname,'..');const port=await freePort();const url=`http://127.0.0.1:${port}`;
  fs.mkdirSync(app.getPath('userData'),{recursive:true});
  const {data,initialResult}=startupOptions({isPackaged:app.isPackaged,resourcesPath:process.resourcesPath,userData:app.getPath('userData')});
  await authorizeProjectDirectory({data,userData:app.getPath('userData'),dialog,t});
  fs.mkdirSync(data,{recursive:true});
  const env={...process.env,SPLAT_PORT:String(port),SPLAT_DATA_DIR:data,SPLAT_SESSION_TOKEN:token,
    HF_HOME:process.env.HF_HOME||path.join(app.getPath('userData'),'models'),
    SPLAT_GEOMETRY_HOME:process.env.SPLAT_GEOMETRY_HOME||path.join(app.getPath('userData'),'geometry-runtime'),
    ...(process.env.SPLAT_GEOMETRY_REPO||app.isPackaged?{SPLAT_GEOMETRY_REPO:process.env.SPLAT_GEOMETRY_REPO||path.join(process.resourcesPath,'geometry','dust3r')}:{}),
    MPLCONFIGDIR:process.env.MPLCONFIGDIR||path.join(app.getPath('userData'),'matplotlib'),MPLBACKEND:process.env.MPLBACKEND||'Agg',
    SPLAT_FRONTEND_DIR:app.isPackaged?path.join(process.resourcesPath,'frontend'):path.join(root,'dist'),PYTHONUNBUFFERED:'1',PYTHONDONTWRITEBYTECODE:'1',PYTHONUTF8:'1'};
  const {command,args,env:runtimeEnv}=backendRuntime({platform:process.platform,isPackaged:app.isPackaged,
    resourcesPath:process.resourcesPath,root,env,exists:fs.existsSync});
  Object.assign(env,runtimeEnv);
  const log=fs.openSync(path.join(app.getPath('userData'),'backend.log'),'a');
  backend=spawn(command,args,{cwd:app.isPackaged?app.getPath('userData'):root,env,stdio:['ignore',log,log],windowsHide:true});
  backend.on('error',error=>{dialog.showErrorBox(t('本地引擎无法启动'),error.message);app.quit();});
  await waitBackend(url);
  session.defaultSession.webRequest.onBeforeSendHeaders({urls:[`${url}/*`]},(details,callback)=>{
    details.requestHeaders['x-splat-token']=token;callback({requestHeaders:details.requestHeaders});
  });
  const win=new BrowserWindow({width:1500,height:990,minWidth:1080,minHeight:740,backgroundColor:'#101319',
    title:'Splat Studio',webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  win.webContents.on('will-navigate',(event,destination)=>{if(!destination.startsWith(url+'/'))event.preventDefault();});
  win.webContents.session.setPermissionRequestHandler((_webContents,_permission,callback)=>callback(false));
  installSceneExport({ipcMain,dialog,win,origin:url,t});
  installLanguageIPC({ipcMain,win,origin:url,preferences});
  const initialURL=new URL(url);if(initialResult)initialURL.searchParams.set('result',initialResult);
  await win.loadURL(initialURL.toString());
}
app.whenReady().then(start).catch(error=>{if(!(error instanceof ProjectAccessCancelled))dialog.showErrorBox(t('Splat Studio 启动失败'),error.message);app.quit();});
app.on('window-all-closed',()=>app.quit());
app.on('before-quit',()=>{if(backend&&!stopping){stopping=true;backend.kill();}});
