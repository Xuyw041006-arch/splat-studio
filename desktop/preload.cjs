const {contextBridge,ipcRenderer}=require('electron');
contextBridge.exposeInMainWorld('splatDesktop',{
  getLanguage:()=>ipcRenderer.invoke('language:get'),
  setLanguage:value=>ipcRenderer.invoke('language:set',value),
  platform:process.platform,version:'0.4.4',
  beginSceneExport:name=>ipcRenderer.invoke('scene-export:begin',name),
  writeSceneExport:(token,text)=>ipcRenderer.invoke('scene-export:write',token,text),
  finishSceneExport:token=>ipcRenderer.invoke('scene-export:finish',token),
  abortSceneExport:token=>ipcRenderer.invoke('scene-export:abort',token),
});
