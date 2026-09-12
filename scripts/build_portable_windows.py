"""Assemble Windows x64 from official binaries and audited wheels, on any OS.

This produces a portable application; it does not claim Windows execution tests.
Downloads are explicit inputs. Every wheel RECORD and dependency is verified.
python scripts/build_portable_windows.py --downloads DIR --wheels DIR --selection JSON --frontend DIR --geometry DIR --output DIR
"""
from pathlib import Path, PurePosixPath
import argparse
import base64
import csv
import email
import hashlib
import io
import json
import shutil
import struct
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
PYTHON_SHA='d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf'
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:raise ValueError(name)
    return path

def unpack(archive,target):
    with zipfile.ZipFile(archive) as z:
        for row in z.infolist():
            safe_name(row.filename)
            z.extract(row,target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['downloads','wheels','selection','frontend','geometry','output']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Output must not exist')
    assert digest(a.downloads/'python.zip')==PYTHON_SHA,'Python SHA mismatch'
    hashes={name.lstrip('*'):value for value,name in (line.split(maxsplit=1) for line in (a.downloads/'electron-sha.txt').read_text().splitlines() if line.strip())}
    assert digest(a.downloads/'electron.zip')==hashes['electron-v38.8.6-win32-x64.zip'],'Electron SHA mismatch'
    wheels={};env={'extra':'','sys_platform':'win32','os_name':'nt','platform_system':'Windows','platform_machine':'AMD64','platform_python_implementation':'CPython','python_version':'3.13','python_full_version':'3.13.15','implementation_name':'cpython','implementation_version':'3.13.15'}
    for name in json.loads(a.selection.read_text()):
        wheel=a.wheels/name
        with zipfile.ZipFile(wheel) as z:
            metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA') and n.count('/')==1)))
            key=canonicalize_name(metadata['Name']);assert key not in wheels
            wheels[key]=(wheel,metadata)
    for wheel,metadata in wheels.values():
        for dep in metadata.get_all('Requires-Dist',[]):
            r=Requirement(dep)
            if r.marker and not r.marker.evaluate(env):continue
            key=canonicalize_name(r.name)
            assert key in wheels and wheels[key][1]['Version'] in r.specifier,(wheel.name,dep)
    unpack(a.downloads/'electron.zip',a.output)
    (a.output/'electron.exe').rename(a.output/'Splat Studio.exe')
    (a.output/'resources/default_app.asar').unlink(missing_ok=True)
    app=a.output/'resources/app';app.mkdir()
    shutil.copytree(ROOT/'desktop',app/'desktop')
    shutil.copyfile(ROOT/'package.json',app/'package.json')
    shutil.copytree(a.frontend,a.output/'resources/frontend')
    shutil.copytree(a.frontend,app/'dist')
    shutil.copytree(a.geometry,a.output/'resources/geometry')
    base=a.output/'resources/backend';python=base/'python'
    unpack(a.downloads/'python.zip',python)
    (python/'python313._pth').write_text('python313.zip\n.\nLib/site-packages\nimport site\n',encoding='utf-8')
    site=python/'Lib/site-packages';site.mkdir(parents=True)
    wheel_manifest=[]
    for wheel,metadata in wheels.values():
        with zipfile.ZipFile(wheel) as z:
            records=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD') and n.count('/')==1)
            # Some upstream Windows wheels use backslashes in RECORD only.
            recorded={row[0].replace('\\','/'):row for row in csv.reader(io.StringIO(z.read(records).decode()))}
            for info in z.infolist():
                name=info.filename;safe_name(name)
                if info.is_dir():continue
                raw=z.read(name);row=recorded.get(name)
                assert row is not None,(wheel.name,name,'unrecorded')
                if row[1]:
                    algorithm,value=row[1].split('=',1)
                    assert algorithm=='sha256'
                    assert base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()==value,(wheel.name,name)
                    assert int(row[2])==len(raw)
                else:assert name==records or name.endswith(('RECORD.jws','RECORD.p7s'))
                parts=PurePosixPath(name).parts
                if parts[0].endswith('.data'):
                    kind=parts[1]
                    target=(site if kind in {'purelib','platlib'} else python if kind=='data' else python/'Scripts' if kind=='scripts' else python/'Include').joinpath(*parts[2:])
                else:target=site.joinpath(*parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert target.read_bytes()==raw,('wheel collision',name)
                target.write_bytes(raw)
        wheel_manifest.append({'name':metadata['Name'],'version':metadata['Version'],'file':wheel.name,'sha256':digest(wheel)})
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_backend import cloud_runtime_sources
    runtime=base/'splat-backend/_internal/cloud-runtime'
    for file in cloud_runtime_sources(ROOT):
        target=runtime/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    for name in ['run_portable.py','windows_self_test.py']:shutil.copyfile(ROOT/'scripts'/name,base/name)
    shutil.copyfile(a.downloads/'VC_redist.x64.exe',a.output/'VC_redist.x64.exe')
    (a.output/'Self-Test.cmd').write_bytes(b'@echo off\r\nchcp 65001 >nul\r\n"%~dp0resources\\backend\\python\\python.exe" -I -B -X utf8 "%~dp0resources\\backend\\windows_self_test.py"\r\nset SPLAT_TEST_EXIT=%ERRORLEVEL%\r\necho.\r\necho Report: %LOCALAPPDATA%\\Splat-Studio-Diagnostics\\self-test.json\r\npause\r\nexit /b %SPLAT_TEST_EXIT%\r\n')
    (a.output/'Install-Runtime.cmd').write_bytes(b'@echo off\r\n"%~dp0VC_redist.x64.exe" /install /passive /norestart\r\npause\r\n')
    pe_count=0;auxiliary_launchers=[]
    for path in a.output.rglob('*'):
        if path.suffix.lower() not in {'.exe','.dll','.pyd'}:continue
        with path.open('rb') as f:
            assert f.read(2)==b'MZ',path
            f.seek(0x3c);offset=struct.unpack('<I',f.read(4))[0]
            f.seek(offset);assert f.read(4)==b'PE\0\0',path
            machine=struct.unpack('<H',f.read(2))[0]
            # Vendor packages ship unused installers/launchers for other CPUs.
            auxiliary=path.name=='VC_redist.x64.exe' or (path.parent==site/'setuptools' and path.name in {'cli.exe','cli-32.exe','cli-arm64.exe','gui.exe','gui-32.exe','gui-arm64.exe'})
            assert machine==0x8664 or auxiliary,(path,machine)
            if machine!=0x8664:auxiliary_launchers.append(path.relative_to(a.output).as_posix())
        pe_count+=1
    manifest={'app_version':json.loads((ROOT/'package.json').read_text())['version'],'platform':'win32-x64','python':'3.13.15','electron':'38.8.6',
              'windows_execution_tested':False,'gpu_training_included':False,
              'checks':['official runtime SHA256','all wheel RECORD hashes','Windows dependency closure','PE architecture'],
              'pe_files':pe_count,'unused_cross_arch_vendor_launchers':auxiliary_launchers,'wheels':wheel_manifest,
              'files':{p.relative_to(a.output).as_posix():digest(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    (a.output/'BUILD-MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(manifest['files']),'pe_files':pe_count,'wheels':len(wheels),'bytes':sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())}))

if __name__=='__main__':main()
