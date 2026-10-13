#!/usr/bin/env python3
"""Efficient post-hoc image-correspondence control for PMC-VQA.

Question/options/gold stay fixed. Only image embeddings are deterministically
reassigned across rows, with different-PMCID replacements enforced whenever a
PMCID can be recovered. Each encoder processes every clean row exactly once;
shuffle variants are then constructed in feature space from separately cached
image and text embeddings. No training/model selection occurs.
"""
from __future__ import annotations

import argparse, csv, importlib.util, json, math, random, re, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image

PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
DEFAULT_SHUFFLE_SEEDS = (42, 7, 123)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-code-dir", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--eval-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--approved-manifest", type=Path, required=True)
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--shuffle-seeds", type=int, nargs="+", default=list(DEFAULT_SHUFFLE_SEEDS))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--expected-biomed-correct", type=int, default=743)
    p.add_argument("--expected-clip-correct", type=int, default=622)
    p.add_argument("--expected-full-correct", type=int, default=760)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_train_module(code_dir: Path) -> Any:
    code_dir = code_dir.resolve()
    candidates = [p for p in [
        code_dir / "79_train_joint_clip_biomed_residual.py",
        code_dir / "79_train_joint_clip_biomed_residual_v1.py",
    ] if p.is_file()]
    if not candidates:
        for name in ("79_train_joint_clip_biomed_residual.py", "79_train_joint_clip_biomed_residual_v1.py"):
            candidates.extend(code_dir.rglob(name))
    unique=[]; seen=set()
    for p in candidates:
        rp=p.resolve(); key=str(rp).lower()
        if key not in seen: seen.add(key); unique.append(rp)
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one residual training module under {code_dir}; found: {unique}")
    path=unique[0]
    spec=importlib.util.spec_from_file_location("joint_residual_current", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Could not import {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod


def pmcid(ref: str):
    m=PMCID_RE.search(str(ref)); return m.group(1).upper() if m else None


def make_cross_group_derangement(rows: Sequence[Mapping[str,Any]], seed:int)->List[int]:
    n=len(rows); rng=random.Random(seed); base=list(range(n))
    if n<2: raise ValueError("Need at least two rows")
    groups=[pmcid(str(r.get("image_ref",""))) for r in rows]
    paths=[str(r["image_path"]) for r in rows]
    for _ in range(20000):
        perm=base.copy(); rng.shuffle(perm)
        if all(i!=j and paths[i]!=paths[j] and (groups[i] is None or groups[j] is None or groups[i]!=groups[j]) for i,j in enumerate(perm)):
            return perm
    for off in range(1,n):
        perm=[(i+off)%n for i in range(n)]
        if all(paths[i]!=paths[j] and (groups[i] is None or groups[j] is None or groups[i]!=groups[j]) for i,j in enumerate(perm)):
            return perm
    raise RuntimeError("Could not construct cross-PMCID derangement")


def write_csv(path:Path, rows:List[Dict[str,Any]]):
    if not rows: return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def exact_mcnemar(clean_correct:torch.Tensor, sh_correct:torch.Tensor):
    fixed=int((~clean_correct & sh_correct).sum().item()); broken=int((clean_correct & ~sh_correct).sum().item())
    n=fixed+broken
    if n==0: p=1.0
    else:
        k=min(fixed,broken); lower=sum(math.comb(n,i) for i in range(k+1))/(2**n); p=min(1.0,2*lower)
    return fixed, broken, p


def accuracy(pred,gold): return float(pred.eq(gold).float().mean().item())


@torch.inference_mode()
def encode_biomed(tm, rows, model, preprocess, tokenizer, device, context_length, progress_every):
    imgs=[]; txts=[]; gold=[]; use_amp=device.type=="cuda"; t0=time.time()
    for idx,row in enumerate(rows):
        with Image.open(row["image_path"]) as raw:
            im=preprocess(raw.convert("RGB")).unsqueeze(0).to(device)
        prompts=tm.make_prompts(row["question"],row["options"])
        tok=tokenizer(prompts,context_length=context_length); tok=tm.move_tokens(tok,device)
        with torch.autocast(device_type=device.type,dtype=torch.float16 if use_amp else torch.float32,enabled=use_amp):
            iv=model.encode_image(im); iv=iv/iv.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            tv=tm.biomed_encode_text(model,tok)
        imgs.append(iv.squeeze(0).float().cpu()); txts.append(tv.float().cpu()); gold.append(int(row["gold_index"]))
        if idx==0 or (idx+1)%progress_every==0: tm.log(f"shuffle_v2 biomed processed={idx+1} elapsed_s={time.time()-t0:.1f}")
    return torch.stack(imgs), torch.stack(txts), torch.tensor(gold,dtype=torch.long)


@torch.inference_mode()
def encode_clip(tm, rows, model, image_processor, tokenizer, device, context_length, progress_every):
    imgs=[]; txts=[]; gold=[]; use_amp=device.type=="cuda"; t0=time.time()
    for idx,row in enumerate(rows):
        with Image.open(row["image_path"]) as raw:
            image_inputs=image_processor(images=raw.convert("RGB"),return_tensors="pt")
        image_inputs=tm.move_mapping(image_inputs,device)
        prompts=tm.make_prompts(row["question"],row["options"])
        text_inputs=tokenizer(prompts,padding=True,truncation=True,max_length=context_length,return_tensors="pt")
        text_inputs=tm.move_mapping(text_inputs,device)
        with torch.autocast(device_type=device.type,dtype=torch.float16 if use_amp else torch.float32,enabled=use_amp):
            iv=tm.normalize_clip_output(model.get_image_features(**image_inputs))
            tv=tm.normalize_clip_output(model.get_text_features(**text_inputs))
        imgs.append(iv.squeeze(0).float().cpu()); txts.append(tv.float().cpu()); gold.append(int(row["gold_index"]))
        if idx==0 or (idx+1)%progress_every==0: tm.log(f"shuffle_v2 clip processed={idx+1} elapsed_s={time.time()-t0:.1f}")
    return torch.stack(imgs), torch.stack(txts), torch.tensor(gold,dtype=torch.long)


def build_payload(rows, image_emb, text_emb, gold, perm=None):
    if perm is None:
        chosen = image_emb
    else:
        chosen = image_emb[torch.tensor(perm, dtype=torch.long)]

    # IMPORTANT: the canonical PMC-VQA baseline scripts compute image-text
    # similarities inside CUDA autocast, i.e. effectively in FP16 on the
    # GTX 1650 setup. Recomputing the same dot-products from cached embeddings
    # in FP32 changes a few near-tied predictions (BiomedCLIP 741 vs 743/2000).
    # Cast to FP16 here to reproduce the frozen canonical scoring protocol
    # exactly while keeping pair features in FP32, as in the training pipeline.
    scores = torch.einsum(
        "nd,nkd->nk",
        chosen.to(torch.float16),
        text_emb.to(torch.float16),
    ).float()
    pair = chosen.unsqueeze(1) * text_emb
    meta=[]
    for r in rows:
        meta.append({"row_index":int(r["row_index"]),"image_ref":r["image_ref"],"image_path":r["image_path"],"question":r["question"],"gold_label":r["gold_label"],
                     "choice_A":r["options"][0],"choice_B":r["options"][1],"choice_C":r["options"][2],"choice_D":r["options"][3]})
    return {"pair_features":pair.contiguous(),"base_scores":scores.contiguous(),"gold_indices":gold.clone(),"rows":meta}


def save_embedding_cache(path: Path, image_emb: torch.Tensor, text_emb: torch.Tensor, gold: torch.Tensor, rows: Sequence[Mapping[str,Any]], encoder: str):
    payload={
        "encoder":encoder,
        "rows":len(rows),
        "image_refs":[str(r["image_ref"]) for r in rows],
        "image_emb":image_emb,
        "text_emb":text_emb,
        "gold":gold,
    }
    torch.save(payload,path)


def load_embedding_cache(path: Path, rows: Sequence[Mapping[str,Any]], encoder: str):
    if not path.is_file(): return None
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        payload=torch.load(path,map_location="cpu")
    refs=[str(r["image_ref"]) for r in rows]
    if payload.get("encoder")!=encoder or int(payload.get("rows",-1))!=len(rows) or payload.get("image_refs")!=refs:
        return None
    return payload["image_emb"],payload["text_emb"],payload["gold"]


def write_status(out_dir: Path, stage: str, **extra):
    payload={"stage":stage,"time":time.strftime("%Y-%m-%d %H:%M:%S"),**extra}
    (out_dir/"run_status.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

def main():
    a=parse_args(); a.repo_code_dir=a.repo_code_dir.resolve(); a.project_root=a.project_root.resolve(); a.eval_csv=a.eval_csv.resolve(); a.image_root=a.image_root.resolve() if a.image_root else None; a.approved_manifest=a.approved_manifest.resolve(); a.out_dir=a.out_dir.resolve(); a.out_dir.mkdir(parents=True,exist_ok=True)
    tm=load_train_module(a.repo_code_dir); device=tm.choose_device(a.device); tm.log(f"image_shuffle_v2_device={device}")
    image_roots=[a.image_root] if a.image_root else tm.discover_image_roots(a.project_root,a.data_root)
    image_roots=[p.resolve() for p in image_roots if p and p.is_dir()]
    idx=tm.index_images(image_roots); rows=tm.parse_rows(a.eval_csv,a.eval_limit,image_roots,idx); n=len(rows)
    manifest=tm.safe_torch_load(a.approved_manifest); states=[Path(p) for p in manifest.get("state_paths",[])]
    if len(states)!=3 or any(not p.is_file() for p in states): raise RuntimeError(f"Invalid frozen state paths: {states}")

    mappings={s:make_cross_group_derangement(rows,s) for s in a.shuffle_seeds}
    map_rows=[]
    for s,perm in mappings.items():
        for i,j in enumerate(perm):
            map_rows.append({"shuffle_seed":s,"row_index":i,"original_image_ref":rows[i]["image_ref"],"replacement_image_ref":rows[j]["image_ref"],"original_pmcid":pmcid(rows[i]["image_ref"]) or "","replacement_pmcid":pmcid(rows[j]["image_ref"]) or ""})
    write_csv(a.out_dir/"image_shuffle_mapping.csv",map_rows)
    write_status(a.out_dir,"mapping_complete",rows=n,seeds=list(a.shuffle_seeds))

    from open_clip import create_model_from_pretrained,get_tokenizer
    bcache=a.out_dir/"biomed_clean_embeddings.pt"
    cached=load_embedding_cache(bcache,rows,"BiomedCLIP")
    if cached is None:
        bm,bprep=create_model_from_pretrained(tm.BIOMED_MODEL_ID); btok=get_tokenizer(tm.BIOMED_MODEL_ID); bm=bm.to(device).eval()
        for p in bm.parameters(): p.requires_grad_(False)
        b_img,b_txt,gold=encode_biomed(tm,rows,bm,bprep,btok,device,a.context_length,a.progress_every); del bm
        save_embedding_cache(bcache,b_img,b_txt,gold,rows,"BiomedCLIP")
    else:
        b_img,b_txt,gold=cached; tm.log(f"loading_cached_shuffle_biomed_embeddings={bcache}")
    write_status(a.out_dir,"biomed_embeddings_complete",rows=n)
    if device.type=="cuda": torch.cuda.empty_cache()

    from transformers import AutoImageProcessor,AutoTokenizer,CLIPModel
    ccache=a.out_dir/"clip_clean_embeddings.pt"
    cached=load_embedding_cache(ccache,rows,"OpenAI CLIP")
    if cached is None:
        cm=CLIPModel.from_pretrained(tm.CLIP_MODEL_ID,local_files_only=a.local_files_only); ctok=AutoTokenizer.from_pretrained(tm.CLIP_MODEL_ID,local_files_only=a.local_files_only); cip=AutoImageProcessor.from_pretrained(tm.CLIP_MODEL_ID,local_files_only=a.local_files_only); cm=cm.to(device).eval()
        for p in cm.parameters(): p.requires_grad_(False)
        clen=int(getattr(cm.config.text_config,"max_position_embeddings",77))
        c_img,c_txt,gold2=encode_clip(tm,rows,cm,cip,ctok,device,clen,a.progress_every); del cm
        save_embedding_cache(ccache,c_img,c_txt,gold2,rows,"OpenAI CLIP")
    else:
        c_img,c_txt,gold2=cached; tm.log(f"loading_cached_shuffle_clip_embeddings={ccache}")
    if not torch.equal(gold,gold2): raise RuntimeError("Gold mismatch between encoders")
    write_status(a.out_dir,"clip_embeddings_complete",rows=n)
    if device.type=="cuda": torch.cuda.empty_cache()

    clean_b=build_payload(rows,b_img,b_txt,gold); clean_c=build_payload(rows,c_img,c_txt,gold)
    clean_preds={"BiomedCLIP":clean_b["base_scores"].argmax(1),"OpenAI CLIP":clean_c["base_scores"].argmax(1)}
    clean_logits=tm.ensemble_logits(states,biomed=clean_b,clip=clean_c,device=device,batch_size=a.batch_size); clean_preds["Full residual ensemble"]=clean_logits.argmax(1)
    clean_counts={k:int(v.eq(gold).sum().item()) for k,v in clean_preds.items()}
    expected={"BiomedCLIP":a.expected_biomed_correct,"OpenAI CLIP":a.expected_clip_correct,"Full residual ensemble":a.expected_full_correct}
    for k in expected:
        if clean_counts[k]!=expected[k]:
            raise RuntimeError(f"Canonical reproduction failed for {k}: observed={clean_counts[k]}/{n}, expected={expected[k]}/{n}. Shuffle results are refused.")
    write_status(a.out_dir,"canonical_reproduction_passed",canonical_reproduction=clean_counts)

    result=[]
    for name,pred in clean_preds.items():
        result.append({"shuffle_seed":"clean","system":name,"rows":n,"correct":clean_counts[name],"accuracy":accuracy(pred,gold),"delta_vs_clean":0.0,"fixed_vs_clean":0,"broken_vs_clean":0,"mcnemar_exact_p":1.0})
    for s,perm in mappings.items():
        sb=build_payload(rows,b_img,b_txt,gold,perm); sc=build_payload(rows,c_img,c_txt,gold,perm)
        preds={"BiomedCLIP":sb["base_scores"].argmax(1),"OpenAI CLIP":sc["base_scores"].argmax(1)}
        slog=tm.ensemble_logits(states,biomed=sb,clip=sc,device=device,batch_size=a.batch_size); preds["Full residual ensemble"]=slog.argmax(1)
        for name,pred in preds.items():
            cc=clean_preds[name].eq(gold); sh=pred.eq(gold); fixed,broken,p=exact_mcnemar(cc,sh); ac=accuracy(pred,gold); ca=accuracy(clean_preds[name],gold)
            result.append({"shuffle_seed":s,"system":name,"rows":n,"correct":int(sh.sum().item()),"accuracy":ac,"delta_vs_clean":ac-ca,"fixed_vs_clean":fixed,"broken_vs_clean":broken,"mcnemar_exact_p":p})

    aggregate=[]
    for name in clean_preds:
        vals=[r for r in result if r["system"]==name and r["shuffle_seed"]!="clean"]
        accs=[float(r["accuracy"]) for r in vals]; dels=[float(r["delta_vs_clean"]) for r in vals]; ma=sum(accs)/len(accs); md=sum(dels)/len(dels); sd=(sum((x-ma)**2 for x in accs)/max(1,len(accs)-1))**0.5
        aggregate.append({"system":name,"clean_accuracy":accuracy(clean_preds[name],gold),"mean_shuffled_accuracy":ma,"sd_shuffled_accuracy":sd,"mean_delta_vs_clean":md,"shuffle_seeds":";".join(map(str,a.shuffle_seeds))})
    write_csv(a.out_dir/"image_shuffle_results.csv",result); write_csv(a.out_dir/"image_shuffle_aggregate.csv",aggregate)
    summary={"analysis":"image_shuffle_control_v4_precision_fixed","status":"posthoc_exploratory_no_training_no_selection","eval_csv":str(a.eval_csv),"rows":n,"shuffle_seeds":a.shuffle_seeds,"canonical_reproduction":clean_counts,"shuffle_rule":"question/options/gold fixed; image embedding replaced; different PMCID enforced whenever PMCID is available","efficiency_note":"each clean image/text pair is encoded once per encoder; shuffle variants are recomposed from separate frozen embeddings", "score_precision_protocol":"base similarities are recomputed in float16 from cached normalized embeddings to match the original CUDA-autocast canonical scoring protocol; pair features remain float32","aggregate":aggregate,"interpretation_rule_predeclared":"A substantial accuracy drop after image shuffling supports genuine dependence on image-question correspondence. A small drop suggests non-visual or benchmark artefacts contribute materially."}
    (a.out_dir/"image_shuffle_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    stale_failure = a.out_dir / "failure_traceback.txt"
    if stale_failure.exists():
        stale_failure.unlink()
    write_status(a.out_dir,"complete",canonical_reproduction=clean_counts)
    print(json.dumps(summary,indent=2))
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception:
        try:
            out=None
            for i,arg in enumerate(sys.argv):
                if arg=="--out-dir" and i+1<len(sys.argv): out=Path(sys.argv[i+1])
            if out is not None:
                out.mkdir(parents=True,exist_ok=True)
                (out/"failure_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
                write_status(out,"failed")
        finally:
            traceback.print_exc()
        raise SystemExit(1)