const fs=require('node:fs');
const path=require('node:path');
const catalog=require('./locales/en.json');
function createLanguagePreferences(filename){
  let language='zh';
  try{const saved=JSON.parse(fs.readFileSync(filename,'utf8'));if(saved.language==='en')language='en';}catch{}
  return {
    get:()=>language,
    set(value){if(value!=='zh'&&value!=='en')throw new Error('Invalid language');
      fs.mkdirSync(path.dirname(filename),{recursive:true});const temporary=filename+'.tmp';
      fs.writeFileSync(temporary,JSON.stringify({language:value})+'\n',{mode:0o600});fs.renameSync(temporary,filename);language=value;return language;
    },
    t(text){return language==='en'?(catalog[text]||text):text;},
  };
}
function installLanguageIPC({ipcMain,win,origin,preferences}){
  function authorize(event){if(event.sender!==win.webContents||event.senderFrame!==win.webContents.mainFrame||new URL(event.senderFrame.url).origin!==origin)throw new Error('Unauthorized language request');}
  ipcMain.handle('language:get',event=>{authorize(event);return preferences.get();});
  ipcMain.handle('language:set',(event,value)=>{authorize(event);return preferences.set(value);});
}
module.exports={createLanguagePreferences,installLanguageIPC};
