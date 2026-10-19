"""Archive and independently check the completed local FLUX validation bundle."""
import argparse,gzip,hashlib,json,struct
from pathlib import Path
from PIL import Image

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text())
def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',rounded))[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted((args.root/'r9700').rglob('report.json'))
    paths=[p for p in paths if p.relative_to(args.root/'r9700').parts[0] in ('quality','nmse','benchmark')]
    assert len(paths)==32,('Expected 18 quality + 10 NMSE + 4 benchmark reports',len(paths))
    entries=[];counts={};alpha=6.93
    expected=[bf16(alpha*(1-i/16)/(1+(alpha-1)*(1-i/16))) for i in range(17)]
    for path in paths:
        raw=path.read_bytes();r=json.loads(raw);purpose=r['measurement_purpose'];counts[purpose]=counts.get(purpose,0)+1
        assert r['arch']=='gfx1201' and r['source']['repo_id']=='black-forest-labs/flux-3-image-klein'
        assert r['settings']['attention_query_chunk']==0 and r['settings']['steps']==16 and r['settings']['vae_dtype']=='float32'
        for i,m in enumerate(r['measurements']):
            assert m['actual_steps']==16 and m['finite_final_latents'] and m['finite_vae_output']
            assert m['timesteps']==[[x] for x in expected[:-1]],('Wrong sampling schedule',path)
            assert m['target_timesteps']==[[x] for x in expected[1:]],('Wrong target schedule',path)
            with Image.open(path.parent/f'run-{i}.png') as im:
                assert sha(im.tobytes())==m['pixel_sha256'] and list(im.size)==m['image_size'] and im.mode==m['image_mode']
        inventory=r['candidate_linear_layers'] if r.get('nmse',{} ) and r['nmse']['candidate']=='exl3' else r['linear_layers']
        layers={k:v for k,v in inventory.items() if v['precision']=='exl3'}
        if layers:
            assert len(layers)==258
            assert all(v['calls']==(0 if n.startswith('mode_blocks.image_cond.') and not r['images'] else 16) for n,v in layers.items())
        if purpose=='nmse':
            assert r['settings']['encoding_cache'] and len(r['nmse']['steps'])==16
            assert all(len(s['latent_sha256'])==64 and len(s['reference_prediction_sha256'])==64 for s in r['nmse']['steps'])
        rel=path.relative_to(args.root/'r9700').as_posix();name=rel.removesuffix('/report.json').replace('/','__')+'.json.gz'
        compressed=gzip.compress(raw,mtime=0);(args.output/name).write_bytes(compressed)
        entries.append({'source_path':rel,'archive':name,'sha256':sha(raw),'compressed_sha256':sha(compressed),'purpose':purpose,
                        'source_commit':r['source_commit'],'images':[m['pixel_sha256'] for m in r['measurements']]})
    assert counts=={'benchmark':4,'nmse':10,'quality':18},counts
    models={}
    for b in (3,4,5):
        folder=args.root/'r9700'/f'exl3-{b}bit';m=read(folder/'quantization.json')
        assert m['complete'] and m['bits']==b and len(m['layers'])==258
        assert m['unquantized_sha256']==sha((folder/'unquantized.safetensors').read_bytes())
        actual=0
        for layer in m['layers'].values():
            p=folder/layer['file'];assert sha(p.read_bytes())==layer['sha256'];actual+=p.stat().st_size
        actual+=(folder/'unquantized.safetensors').stat().st_size
        (args.output/f'model-{b}bit.json.gz').write_bytes(gzip.compress((folder/'quantization.json').read_bytes(),mtime=0))
        models[str(b)]={'effective_bpw':m['effective_bits_per_source_element'],'safetensors_bytes':actual,
                        'retained_sha256':m['unquantized_sha256'],'calibration':m['calibration'],'layer_count':258}
    assert len({m['retained_sha256'] for m in models.values()})==1
    assert len({tuple(m['calibration']) for m in models.values()})==1
    (args.output/'index.json').write_text(json.dumps({'reports':entries,'counts':counts,'models':models,
        'all_png_payloads_match_reports':True,'official_shift_6_93_schedule_verified':True,
        'all_checkpoint_file_hashes_verified':True},indent=2)+'\n')
    print(json.dumps({'counts':counts,'models':models},indent=2))
if __name__=='__main__':main()
