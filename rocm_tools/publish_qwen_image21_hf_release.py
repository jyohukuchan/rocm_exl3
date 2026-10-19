"""Explicitly upload one prepared EXL3 package, verify the remote files and tag it."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, ModelCard


def digest(path, algorithm='sha256', git_blob=False):
    h=hashlib.new(algorithm)
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    publish(args)


def publish(args):
    root=args.folder.resolve()
    publication=json.loads((root/'PUBLICATION.json').read_text())
    assert publication['repository']==args.repo_id
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text())
    entries={e['path']:e for e in manifest['files']}
    expected=set(entries)|{'FILE_MANIFEST.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==expected, 'Staging contains unexpected or missing files'
    for name,entry in entries.items():
        p=root/name
        assert p.resolve().is_relative_to(root)
        assert p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256'], name
    ModelCard.load(str(root/'README.md')).validate()
    api=HfApi()
    assert api.whoami()['name']==args.repo_id.split('/')[0], 'Publishing account differs from destination'
    if api.repo_exists(args.repo_id,repo_type='model'):
        if not args.resume:
            raise FileExistsError('Repository already exists; inspect it before explicit --resume')
        remote=Path(hf_hub_download(args.repo_id,'FILE_MANIFEST.json',force_download=True))
        assert remote.read_bytes()==(root/'FILE_MANIFEST.json').read_bytes(), 'Existing repository is a different release'
    else:
        api.create_repo(args.repo_id,repo_type='model',private=False)
    print('Upload license, model card, and evaluation evidence:',args.repo_id,flush=True)
    api.upload_folder(repo_id=args.repo_id,repo_type='model',folder_path=root,
                      allow_patterns=[p for p in expected if '/' not in p],
                      commit_message='Add EXL3 model card, license, provenance and NMSE evaluation')
    # Stay below a 500-LFS-file commit by publishing one bit width at a time.
    for bits in (3,4,5):
        print(f'Upload {bits}bit checkpoint: {args.repo_id}',flush=True)
        api.upload_folder(repo_id=args.repo_id,repo_type='model',folder_path=root/f'{bits}bit',
                          path_in_repo=f'{bits}bit',commit_message=f'Publish evaluated EXL3 {bits}bit DiT checkpoint')
    info=api.model_info(args.repo_id,files_metadata=True)
    assert info.private is False
    remote={f.rfilename:f for f in info.siblings}
    assert set(remote)-{'.gitattributes'}==expected
    for name in sorted(expected):
        entry=remote[name];local=root/name
        assert entry.size==local.stat().st_size, name
        if entry.lfs is not None:
            oid=entry.lfs.sha256 if hasattr(entry.lfs,'sha256') else entry.lfs.get('sha256',entry.lfs.get('oid'))
            assert oid==digest(local), name
        else:
            assert entry.blob_id==digest(local,'sha1',git_blob=True), name
    tag=publication['release_tag']
    api.create_tag(args.repo_id,tag=tag,revision=info.sha,repo_type='model',exist_ok=True)
    # Verify visibility without the uploader's credentials and pin the release tag.
    public=HfApi(token=False).model_info(args.repo_id,revision=tag,files_metadata=True)
    assert public.private is False and public.sha==info.sha
    remote_manifest=Path(hf_hub_download(args.repo_id,'FILE_MANIFEST.json',revision=tag,token=False,force_download=True))
    assert remote_manifest.read_bytes()==(root/'FILE_MANIFEST.json').read_bytes()
    result={'repo_id':args.repo_id,'url':f'https://huggingface.co/{args.repo_id}',
            'revision':info.sha,'release_tag':tag,'public':True,
            'verified_files':len(expected),'remote_file_hashes_match':True,
            'bytes':sum((root/name).stat().st_size for name in expected)}
    args.result.parent.mkdir(parents=True,exist_ok=True)
    args.result.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
