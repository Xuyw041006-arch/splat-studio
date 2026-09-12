"""Validate an App task before loading any task source. No Colab dependency."""
from pathlib import Path, PurePosixPath
import zipfile, hashlib, stat, json, shutil

def unpack_task(package, destination):
    """在执行包内代码前检查大小、路径和每个文件的 SHA256。"""
    package, destination = Path(package), Path(destination)
    if destination.exists(): raise ValueError('解压目录必须是新目录。')
    if package.stat().st_size > 4*1024**3: raise ValueError('任务包超过 4 GiB。')
    with zipfile.ZipFile(package) as z:
        infos=z.infolist(); seen=set(); total=0
        if len(infos)>20001: raise ValueError('任务文件过多。')
        for info in infos:
            name=info.filename; parts=name.split('/'); mode=info.external_attr>>16
            if (not name or len(name)>512 or any(c in name for c in ('\\',':','\x00'))
                or PurePosixPath(name).is_absolute() or any(p in ('','.','..') for p in parts)
                or name.casefold() in seen or info.is_dir() or info.flag_bits&1
                or stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
                raise ValueError('不支持的任务 ZIP 路径或文件类型。')
            seen.add(name.casefold()); total+=info.file_size
            if info.file_size>512*1024**2 or total>4*1024**3: raise ValueError('解压内容过大。')
        if 'manifest.json' not in seen: raise ValueError('请选择 App 导出的任务 ZIP，不是确认包、结果包或源码包。')
        if z.getinfo('manifest.json').file_size>8*1024**2: raise ValueError('清单过大。')
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('format')!='splat-studio-cloud-task/1':
            raise ValueError('请选择 App 导出的任务 ZIP，不是源码 ZIP、确认包或结果包。')
        declared=set()
        for row in manifest['files']:
            name=row['path']
            if name in declared or name=='manifest.json': raise ValueError('任务清单有重复。')
            declared.add(name)
            if z.getinfo(name).file_size!=row['bytes']: raise ValueError('文件长度不匹配。')
            with z.open(name) as f: value=hashlib.file_digest(f,'sha256').hexdigest()
            if value!=row['sha256']: raise ValueError('文件校验失败：'+name)
        if declared|{'manifest.json'}!={i.filename for i in infos}: raise ValueError('任务清单与内容不一致。')
        destination.mkdir()
        for info in infos:
            target=destination/info.filename; target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src, target.open('xb') as dst: shutil.copyfileobj(src,dst)
    return manifest

