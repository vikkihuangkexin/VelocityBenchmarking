"""Serialize completely, fsync, atomically replace, and verify every CSV byte."""
from pathlib import Path
import os,tempfile,hashlib
def write_bytes_verified(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.complete-',suffix=path.suffix,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as h:
            h.write(data);h.flush();os.fsync(h.fileno())
        assert Path(name).read_bytes()==data,'Temporary output byte mismatch'
        os.replace(name,path)
        assert path.read_bytes()==data,'Published output byte mismatch'
    finally:
        Path(name).unlink(missing_ok=True)
def save_csv(frame,path,index=False,**kwargs):
    data=frame.to_csv(None,index=index,na_rep='NA',**kwargs).encode('utf-8')
    write_bytes_verified(path,data)
