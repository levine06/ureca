"""Candidate environment, accessibility errors and the scoring / controls command-line tools, on synthetic structures."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
PB = pytest.importorskip("PeptideBuilder")

from accessfold import AtomicStructure
from accessfold.scoring import candidate_profile, restrict_environment, truth_profile
from accessfold.structure_metrics import ca_coordinates, structural_metrics
from accessfold.structures.mmcif import load_chain_from_mmcif
from accessfold.structures.predicted import load_predicted_chain
from accessfold.synthetic import smooth_displacement, write_chain_cif
from accessfold.truth import truth_arrays

# helix - turn - helix - turn - helix: compact enough for a spread of buried / exposed residues
SEQ = "MKTAYIAKQRQISFVKSHFSRQ" + "GDGS" + "LEEALKKAAEELLKRH" + "GNGS" + "VKELIEKAKRLAE"
PHI_PSI = ([(-57, -47)] * 22 + [(-80, 0), (80, 0), (-80, 0), (-100, 130)] + [(-57, -47)] * 16 + [(-80, 0), (80, 0), (-80, 0), (-100, 130)]
           + [(-57, -47)] * 13)
RADII = (1.4, 2.5, 4.0, 6.0)


def full_structure():
    s = PB.make_structure(SEQ, [p for p, _ in PHI_PSI[:-1]], [q for _, q in PHI_PSI[:-1]])
    xyz, el, nm, ri = [], [], [], []
    for i, res in enumerate(s[0]["A"]):
        for a in res:
            if a.element != "H":
                xyz.append(a.coord)
                el.append(a.element.upper())
                nm.append(a.get_name())
                ri.append(i)
    from accessfold.structures.mmcif import ONE_TO_THREE
    return AtomicStructure(np.array(xyz), el, nm, ri, np.array([ONE_TO_THREE[c] for c in SEQ]))


def truth_from(full):
    """The experimental version: residues 8-9 unresolved, residue 5 loses its side chain, residue 1 is selenomethionine."""
    keep = ~np.isin(full.residue_index, [7, 8])
    keep &= ~((full.residue_index == 4) & ~np.isin(full.atom_names, ["N", "CA", "C", "O", "CB"]))
    names = np.where((full.residue_index == 0) & (full.atom_names == "SD"), "SE", full.atom_names)
    elems = np.where(names == "SE", "SE", full.elements)
    return AtomicStructure(full.coords[keep], elems[keep], names[keep], full.residue_index[keep], full.residue_names)


@pytest.fixture(scope="module")
def structures(tmp_path_factory):
    full = full_structure()
    truth = truth_from(full)
    d = tmp_path_factory.mktemp("sc")
    write_chain_cif(d / "TEST.cif.gz", truth)
    return full, truth, d


def test_restrict_environment_mirrors_truth_atoms(structures):
    full, truth, _ = structures
    env = restrict_environment(full, truth, "truth_atoms")
    same = lambda s: sorted(zip(s.residue_index.tolist(), np.where(s.atom_names == "SE", "SD", s.atom_names).tolist()))  # noqa: E731
    assert same(env) == same(truth)                                          # SE <-> SD matched, gap and truncation copied
    assert restrict_environment(full, truth, "all_atoms").n_atoms == full.n_atoms
    res = restrict_environment(full, truth, "truth_residues")
    assert not res.resolved_mask[[7, 8]].any() and res.resolved_mask[4] and (res.residue_index == 4).sum() == (full.residue_index == 4).sum()
    with pytest.raises(ValueError):
        restrict_environment(full, truth, "bogus")


def test_identical_candidate_has_zero_error_only_in_matched_environment(structures):
    full, truth, d = structures
    s, complete, qc = load_chain_from_mmcif(d / "TEST.cif.gz", "A", SEQ)
    t = truth_arrays(s, complete, RADII, modified=qc["modified_residues"])
    y = truth_profile(t, RADII)
    assert not t["mask"][[4, 7, 8]].any() and t["mask"][0]                   # truncated + unresolved masked, MSE kept
    matched = candidate_profile(full, s, t, RADII, 1000, "truth_atoms")
    assert np.array_equal(np.isnan(matched), np.isnan(y))
    assert np.nanmax(np.abs(matched - y)) < 0.05                             # only Se vs S radius differs
    loose = candidate_profile(full, s, t, RADII, 1000, "all_atoms")
    assert np.nanmax(np.abs(loose - y)) > 0.1                                # complete candidate near the gap disagrees
    assert np.nanmax(np.abs(loose - y)) > 3 * np.nanmax(np.abs(matched - y))


def test_bent_candidates_error_grows_with_displacement(structures):
    full, truth, d = structures
    s, complete, qc = load_chain_from_mmcif(d / "TEST.cif.gz", "A", SEQ)
    t = truth_arrays(s, complete, RADII, modified=qc["modified_residues"])
    from accessfold.controls import accessibility_error, spearman
    t_ca, t_has = ca_coordinates(s)
    sig = np.geomspace(0.3, 6.0, 12)
    cands, rmsd = [], []
    for i, sg in enumerate(sig):
        xyz = smooth_displacement(full.coords, t_ca[t_has], float(sg), seed=100 + i)
        cand = AtomicStructure(xyz, full.elements, full.atom_names, full.residue_index, full.residue_names)
        cands.append(candidate_profile(cand, s, t, RADII, 1000, "truth_atoms"))
        c_ca, c_has = ca_coordinates(cand)
        rmsd.append(structural_metrics(t_ca, t_has, c_ca, c_has)["rmsd_ca"])
    assert np.all(np.diff(rmsd) > 0) and rmsd[0] < 1.0 and rmsd[-1] > 3.0
    e = accessibility_error(truth_profile(t, RADII), np.stack(cands), metric="mae")
    assert spearman(e, np.array(rmsd)) > 0.7 and e[0] < e[-1]


def test_load_predicted_pdb_and_cif_agree(structures, tmp_path):
    full, truth, d = structures
    lines = []
    for i in range(full.n_atoms):
        x, y, z = full.coords[i]
        el = full.elements[i]
        lines.append(f"ATOM  {i + 1:5d} {full.atom_names[i]:<4s} {full.residue_names[full.residue_index[i]]} A{full.residue_index[i] + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n")
    (tmp_path / "p.pdb").write_text("".join(lines) + "END\n")
    write_chain_cif(tmp_path / "p.cif", full)
    a, comp_a, _ = load_predicted_chain(tmp_path / "p.pdb", "A", SEQ)
    b, comp_b, _ = load_predicted_chain(tmp_path / "p.cif", "A", SEQ)
    assert a.n_atoms == b.n_atoms == full.n_atoms and comp_a.all() and comp_b.all()
    assert np.allclose(a.coords, full.coords, atol=1e-3) and np.allclose(b.coords, full.coords, atol=1e-3)
    with pytest.raises(ValueError, match="numbered 1..N"):
        load_predicted_chain(tmp_path / "p.pdb", "A", SEQ[:20])
    with pytest.raises(ValueError, match="unsupported"):
        load_predicted_chain(tmp_path / "p.xyz", "A", SEQ)


def run(args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_cli_pipeline_end_to_end(structures, tmp_path):
    full, truth, d = structures
    root = Path(__file__).resolve().parents[1] / "scripts"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "TEST.cif.gz").write_bytes((d / "TEST.cif.gz").read_bytes())
    ds = tmp_path / "ds.csv"
    ds.write_text("pdb_id,chain_id,label_chain_id,sequence,length,resolution,resolved_fraction,training_or_test_split,release_date\n"
                  f"TEST,A,A,{SEQ},{len(SEQ)},1.5,0.9,development,2000-01-01\n")
    r = run([root / "compute_truth_accessibility.py", "--csv", ds, "--cache-dir", cache, "--out", tmp_path / "truth", "--offline",
             "--workers", 1, "--radii", *RADII])
    assert r.returncode == 0, r.stdout + r.stderr
    # candidates: the complete chain bent by growing amounts (written as .cif.gz), with a deliberately imperfect ranking
    t_ca, t_has = ca_coordinates(load_chain_from_mmcif(cache / "TEST.cif.gz", "A", SEQ)[0])
    sig = np.geomspace(0.3, 6.0, 12)
    rows = [("entry", "candidate", "score")]
    for i, sg in enumerate(sig):
        write_chain_cif(tmp_path / "preds" / "TEST_A" / f"c{i:02d}.cif.gz", full,
                        smooth_displacement(full.coords, t_ca[t_has], float(sg), seed=200 + i))
        rows.append(("TEST_A", f"c{i:02d}.cif.gz", -float(sg) + (0.5 if i == 3 else 0.0)))
    (tmp_path / "ranking.csv").write_text("\n".join(",".join(map(str, x)) for x in rows) + "\n")
    scores = tmp_path / "scores"
    r = run([root / "score_candidates.py", "--csv", ds, "--truth-dir", tmp_path / "truth", "--cache-dir", cache, "--candidates",
             str(tmp_path / "preds" / "{entry}" / "*.cif.gz"), "--ranking-csv", tmp_path / "ranking.csv", "--out", scores,
             "--workers", 1])
    assert r.returncode == 0, r.stdout + r.stderr
    cs = list(csv.DictReader(open(scores / "candidate_scores.csv")))
    es = list(csv.DictReader(open(scores / "entry_summary.csv")))
    assert len(cs) == 12 and len(es) == 1
    rm = np.array([float(x["rmsd_ca"]) for x in cs])
    assert np.all(np.diff(rm) > 0) and {"ac_error", "ac_corr_error", "ac_mae_1.4", "ac_mae_6", "lddt_ca", "tm_score"} <= set(cs[0])
    e = es[0]
    assert int(e["n_candidates"]) == 12 and float(e["rmsd_spread"]) > 4 and float(e["rho_ac_vs_rmsd"]) > 0.7
    assert float(e["rmsd_top1"]) == pytest.approx(rm[3], abs=1e-6)             # the highest score was candidate 3
    assert float(e["rmsd_min"]) == pytest.approx(rm[0], abs=1e-6) and float(e["top1_minus_best"]) > 0
    z = np.load(scores / "TEST_A.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["environment"] == "truth_atoms" and meta["n_residues_accessibility"] < len(SEQ) and z["c"].shape[0] == 12
    # resumable
    before = (scores / "TEST_A.npz").stat().st_mtime_ns
    run([root / "score_candidates.py", "--csv", ds, "--truth-dir", tmp_path / "truth", "--cache-dir", cache, "--candidates",
         str(tmp_path / "preds" / "{entry}" / "*.cif.gz"), "--out", scores, "--workers", 1])
    assert (scores / "TEST_A.npz").stat().st_mtime_ns == before
    # missing candidates are reported, not fatal to other entries
    r = run([root / "score_candidates.py", "--csv", ds, "--truth-dir", tmp_path / "truth", "--cache-dir", cache, "--candidates",
             str(tmp_path / "nowhere" / "{entry}" / "*.cif"), "--out", tmp_path / "scores2", "--workers", 1])
    assert "no candidates match" in (tmp_path / "scores2" / "failures.csv").read_text()
    # controls on the one protein (needs >= min-candidates; a single protein is enough for the plumbing)
    r = run([root / "run_controls.py", "--scores-dir", scores, "--out", tmp_path / "ctl", "--n-perm", 60, "--n-boot", 200,
             "--min-candidates", 8, "--n-other", 3, "--controls", "real", "shuffle", "reverse", "dropout:0.5", "binary:0.2"])
    assert r.returncode == 0, r.stdout + r.stderr
    summ = list(csv.DictReader(open(tmp_path / "ctl" / "controls_summary.csv")))
    real = [x for x in summ if x["control"] == "real" and x["metric"] == "rmsd_ca" and x["error_metric"] == "mae"][0]
    assert float(real["median_rho"]) > 0.7 and real["p_median_vs_shuffle"] != ""
    assert {"mae", "corr"} == {x["error_metric"] for x in summ}


# ============================================================================== review fixes: bookkeeping, completeness,
# correspondence, ranking, exit status / staleness, Windows csv limit
import importlib.util  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipeline(structures, tmp_path_factory):
    """Truth files for one entry plus six graded candidates (c00 closest ... c05 furthest)."""
    full, truth, d = structures
    tmp = tmp_path_factory.mktemp("pipe")
    cache = tmp / "cache"
    cache.mkdir()
    (cache / "TEST.cif.gz").write_bytes((d / "TEST.cif.gz").read_bytes())
    ds = tmp / "ds.csv"
    ds.write_text("pdb_id,chain_id,label_chain_id,sequence,length,resolution,resolved_fraction,training_or_test_split,release_date\n"
                  f"TEST,A,A,{SEQ},{len(SEQ)},1.5,0.9,development,2000-01-01\n")
    r = run([ROOT / "compute_truth_accessibility.py", "--csv", ds, "--cache-dir", cache, "--out", tmp / "truth", "--offline",
             "--workers", 1, "--radii", *RADII])
    assert r.returncode == 0, r.stdout + r.stderr
    t_ca, t_has = ca_coordinates(load_chain_from_mmcif(cache / "TEST.cif.gz", "A", SEQ)[0])
    for i, sg in enumerate(np.geomspace(0.3, 5.0, 6)):
        write_chain_cif(tmp / "preds" / "TEST_A" / f"c{i:02d}.cif.gz", full,
                        smooth_displacement(full.coords, t_ca[t_has], float(sg), seed=300 + i))
    return {"tmp": tmp, "cache": cache, "ds": ds, "truth": tmp / "truth", "preds": tmp / "preds", "full": full}


def task(p, out, pattern=None, **kw):
    s = load_script("score_candidates")
    row = s.read_rows([p["ds"]])[0]
    return s, (row, str(kw.get("truth", p["truth"])), str(kw.get("cache", p["cache"])), pattern or str(p["preds"] / "{entry}" / "*.cif.gz"),
               "A", str(out), kw.get("environment", "truth_atoms"), kw.get("kind", "rel"), kw.get("overwrite", False),
               kw.get("min_candidates", 2), kw.get("min_coverage", 1.0), kw.get("verify_truth", False))


def test_failure_after_metrics_cannot_misalign_scores_and_files(pipeline, tmp_path, monkeypatch):
    import accessfold.scoring as sc
    real = sc.candidate_profile
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                                   # the 3rd candidate (c02) dies AFTER its structural metrics
            raise RuntimeError("simulated accessibility failure")
        return real(*a, **k)

    monkeypatch.setattr(sc, "candidate_profile", flaky)
    out = tmp_path / "o"
    out.mkdir()
    s, t = task(pipeline, out)
    name, err, status = s.process_entry(t)
    assert err is None, err
    z = np.load(out / "TEST_A.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["candidates"] == ["c00.cif.gz", "c01.cif.gz", "c03.cif.gz", "c04.cif.gz", "c05.cif.gz"]
    assert any("c02.cif.gz" in f and "simulated" in f for f in meta["failed_candidates"])
    seq_truth, _, _ = load_chain_from_mmcif(pipeline["cache"] / "TEST.cif.gz", "A", SEQ)
    t_ca, t_has = ca_coordinates(seq_truth)
    for i, nm in enumerate(meta["candidates"]):               # every stored value belongs to the file next to it
        cand, _, _ = load_predicted_chain(pipeline["preds"] / "TEST_A" / nm, "A", SEQ)
        c_ca, c_has = ca_coordinates(cand)
        m = structural_metrics(t_ca, t_has, c_ca, c_has)
        assert z["rmsd_ca"][i] == pytest.approx(m["rmsd_ca"]) and z["tm_score"][i] == pytest.approx(m["tm_score"])
    assert len(z["rmsd_ca"]) == len(z["lddt_ca"]) == len(z["ac_mae"]) == len(z["c"]) == len(z["coverage_ca"]) == 5


def write_cut(path, full, drop_residues=(), drop_sidechain_of=()):
    keep = ~np.isin(full.residue_index, list(drop_residues))
    keep &= ~(np.isin(full.residue_index, list(drop_sidechain_of)) & ~np.isin(full.atom_names, ["N", "CA", "C", "O", "CB"]))
    cut = AtomicStructure(full.coords[keep], full.elements[keep], full.atom_names[keep], full.residue_index[keep], full.residue_names)
    write_chain_cif(path, cut)


def test_incomplete_candidates_are_rejected_unless_coverage_is_relaxed(pipeline, tmp_path):
    full = pipeline["full"]
    preds = tmp_path / "preds" / "TEST_A"
    for f in (pipeline["preds"] / "TEST_A").glob("*.cif.gz"):
        (preds).mkdir(parents=True, exist_ok=True)
        (preds / f.name).write_bytes(f.read_bytes())
    write_cut(preds / "gap.cif.gz", full, drop_residues=range(30, 40))             # a whole region missing
    write_cut(preds / "trunc.cif.gz", full, drop_sidechain_of=[15, 16, 17])        # side chains missing in the mask
    pattern = str(tmp_path / "preds" / "{entry}" / "*.cif.gz")
    out = tmp_path / "o1"
    out.mkdir()
    s, t = task(pipeline, out, pattern)
    assert s.process_entry(t)[1] is None
    meta = json.loads(str(np.load(out / "TEST_A.npz")["meta"]))
    assert "gap.cif.gz" not in meta["candidates"] and "trunc.cif.gz" not in meta["candidates"] and len(meta["candidates"]) == 6
    reasons = " ".join(meta["failed_candidates"])
    assert "gap.cif.gz" in reasons and "trunc.cif.gz" in reasons and "incomplete candidate" in reasons
    # relaxed coverage: they are scored but flagged, and the controls loader still drops them by default
    out2 = tmp_path / "o2"
    out2.mkdir()
    s, t = task(pipeline, out2, pattern, min_coverage=0.5)
    assert s.process_entry(t)[1] is None
    z = np.load(out2 / "TEST_A.npz")
    meta2 = json.loads(str(z["meta"]))
    assert len(meta2["candidates"]) == 8
    i_gap, i_tr = meta2["candidates"].index("gap.cif.gz"), meta2["candidates"].index("trunc.cif.gz")
    assert z["coverage_ca"][i_gap] < 0.95 and z["coverage_mask"][i_tr] < 0.95 and z["coverage_ca"][0] == 1.0
    from accessfold.controls import load_entry
    assert len(load_entry(out2 / "TEST_A.npz").rmsd_ca) == 6
    assert len(load_entry(out2 / "TEST_A.npz", min_coverage=0.5).rmsd_ca) == 8


def test_predicted_chain_correspondence_is_verified(pipeline, tmp_path):
    full = pipeline["full"]

    def pdb_lines(mod=None):
        out = []
        for i in range(full.n_atoms):
            p = int(full.residue_index[i])
            rn, num, ic, nm = full.residue_names[p], p + 1, " ", full.atom_names[i]
            if mod:
                rn, num, ic, nm = mod(i, p, rn, num, ic, nm)
            x, y, z = full.coords[i]
            out.append(f"ATOM  {i + 1:5d} {nm:<4s} {rn:>3s} A{num:4d}{ic}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {full.elements[i]:>2s}\n")
        return "".join(out)

    ok = tmp_path / "ok.pdb"
    ok.write_text(pdb_lines())
    assert load_predicted_chain(ok, "A", SEQ)[0].n_atoms == full.n_atoms
    bad = {
        "wrong_name": (lambda i, p, rn, num, ic, nm: ("GLY" if p == 10 else rn, num, ic, nm), "sequence has"),
        "insertion": (lambda i, p, rn, num, ic, nm: (rn, num, "A" if p == 12 else ic, nm), "insertion code"),
        "shifted": (lambda i, p, rn, num, ic, nm: (rn, num + 1, ic, nm), "does not fit|sequence has"),
    }
    for label, (mod, msg) in bad.items():
        f = tmp_path / f"{label}.pdb"
        f.write_text(pdb_lines(mod))
        with pytest.raises(ValueError, match=msg):
            load_predicted_chain(f, "A", SEQ)
    dup = tmp_path / "dup.pdb"                                   # alternate-location style duplicate of one atom
    first = pdb_lines().splitlines(keepends=True)
    dup.write_text("".join(first[:5] + [first[4]] + first[5:]))
    with pytest.raises(ValueError, match="duplicate atom"):
        load_predicted_chain(dup, "A", SEQ)
    # mmCIF candidate of a DIFFERENT sequence (right length): identities differ -> rejected, not scored
    other = "".join("A" if c != "A" else "G" for c in SEQ)
    swapped = AtomicStructure(full.coords, full.elements, full.atom_names, full.residue_index,
                              np.array([{"A": "ALA", "G": "GLY"}[c] for c in other]))
    write_chain_cif(tmp_path / "swap.cif", swapped)
    with pytest.raises(ValueError, match="residue identities differ"):
        load_predicted_chain(tmp_path / "swap.cif", "A", SEQ)
    with pytest.raises(ValueError, match="no polymer atoms"):
        load_predicted_chain(pipeline["preds"] / "TEST_A" / "c00.cif.gz", "Z", SEQ)


def test_ranking_manifest_and_partial_ranking(pipeline, tmp_path):
    s = load_script("score_candidates")
    out = tmp_path / "o"
    out.mkdir()
    _, t = task(pipeline, out)
    assert s.process_entry(t)[1] is None
    scores = {f"c{i:02d}": -float(i) for i in range(6)}
    scores["c03"] = 99.0                                                       # make c03 the top-ranked candidate

    def manifest(rows, header):
        p = tmp_path / "m.csv"
        p.write_text(",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
        return p

    # a manifest with different column names, directory-qualified file names (Windows separators) and pdb_id as the entry
    m = manifest([("TEST", f"runs\\TEST\\{k}.cif.gz", v, "x") for k, v in scores.items()], ["pdb_id", "structure_path", "sample_ranking_score", "note"])
    s.collect(out, ranking=s.read_ranking(m))
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    rm = {r["candidate"]: float(r["rmsd_ca"]) for r in csv.DictReader(open(out / "candidate_scores.csv"))}
    assert e["ranking_status"] == "complete" and float(e["rmsd_top1"]) == pytest.approx(rm["c03.cif.gz"])
    assert float(e["rmsd_best_top5"]) == pytest.approx(min(rm[f"c0{i}.cif.gz"] for i in (3, 0, 1, 2, 4)))
    assert e["n_tied_at_top1"] == "1"
    # a `rank` column (1 = best) works too
    m = manifest([("TEST_A", f"{k}.cif.gz", 1 if k == "c05" else 2 + i) for i, k in enumerate(scores) if k != "c05"] + [("TEST_A", "c05.cif.gz", 1)],
                 ["entry", "candidate", "rank"])
    s.collect(out, ranking=s.read_ranking(m))
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    assert float(e["rmsd_top1"]) == pytest.approx(rm["c05.cif.gz"])
    # PARTIAL ranking: one scored candidate has no score -> no top-N numbers are reported at all
    m = manifest([("TEST_A", f"{k}.cif.gz", v) for k, v in scores.items() if k != "c04"], ["entry", "candidate", "score"])
    s.collect(out, ranking=s.read_ranking(m))
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    assert e["ranking_status"].startswith("incomplete") and e["rmsd_top1"] == "" and e["rmsd_best_top5"] == ""
    # ambiguous or missing columns are an error that names the fix, never a silent guess
    m = manifest([("TEST_A", "c00.cif.gz", 1, 2)], ["entry", "candidate", "score", "ranking_score"])
    with pytest.raises(SystemExit, match="ranking-columns score="):
        s.read_ranking(m)
    assert s.read_ranking(m, {"score": "ranking_score"}).records[0]["score"] == 2.0
    m = manifest([("TEST_A", "c00.cif.gz", 1)], ["foo", "bar", "baz"])
    with pytest.raises(SystemExit, match="ranking-columns entry="):
        s.read_ranking(m)


def test_exit_status_failures_file_and_stale_results(pipeline, tmp_path):
    out = tmp_path / "o"
    base = [ROOT / "score_candidates.py", "--csv", pipeline["ds"], "--truth-dir", pipeline["truth"], "--cache-dir", pipeline["cache"],
            "--out", out, "--workers", 1]
    good = str(pipeline["preds"] / "{entry}" / "*.cif.gz")
    r = run(base + ["--candidates", str(tmp_path / "nowhere" / "{entry}" / "*.cif")])
    assert r.returncode == 1 and "no candidates match" in (out / "failures.csv").read_text()      # scheduler sees the failure
    r = run(base + ["--candidates", good])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (out / "failures.csv").exists()                                                        # stale failure list removed
    assert "scored TEST_A" in r.stdout
    r = run(base + ["--candidates", good])
    assert "reused TEST_A" in r.stdout
    r = run(base + ["--candidates", good, "--environment", "all_atoms"])                               # settings changed
    assert "stale->rescored TEST_A" in r.stdout
    assert json.loads(str(np.load(out / "TEST_A.npz")["meta"]))["environment"] == "all_atoms"
    # a candidate added later, or changed in place, invalidates the stored result
    extra = tmp_path / "preds2" / "TEST_A"
    extra.mkdir(parents=True)
    for f in (pipeline["preds"] / "TEST_A").glob("*.cif.gz"):
        (extra / f.name).write_bytes(f.read_bytes())
    good2 = str(tmp_path / "preds2" / "{entry}" / "*.cif.gz")
    run(base + ["--candidates", good2, "--environment", "all_atoms"])
    write_chain_cif(extra / "c06.cif.gz", pipeline["full"], pipeline["full"].coords + 0.5)
    r = run(base + ["--candidates", good2, "--environment", "all_atoms"])
    assert "stale->rescored TEST_A" in r.stdout and len(json.loads(str(np.load(out / "TEST_A.npz")["meta"]))["candidates"]) == 7
    write_chain_cif(extra / "c06.cif.gz", pipeline["full"], pipeline["full"].coords + 0.7)         # same name, new content
    r = run(base + ["--candidates", good2, "--environment", "all_atoms"])
    assert "stale->rescored TEST_A" in r.stdout


def test_scripts_survive_windows_csv_limit(monkeypatch):
    import csv as csvmod
    real = csvmod.field_size_limit

    def windows_like(n=None):                      # a C long is 32 bit on Windows: sys.maxsize overflows
        if n is not None and n > 2 ** 31 - 1:
            raise OverflowError("Python int too large to convert to C long")
        return real(n) if n is not None else real()

    monkeypatch.setattr(csvmod, "field_size_limit", windows_like)
    for name in ("compute_truth_accessibility", "score_candidates", "make_synthetic_candidates"):
        load_script(name)                          # raised OverflowError at import before the fix
    assert real() > 100_000                        # and the limit was actually raised, not left at the 131072 default


# ============================================================================== second review: glycine side chains, stale results,
# original ranks, min_candidates on reuse, ranking collisions, truth/cache identity, side-chain CSV export
def read_meta(path):
    return json.loads(str(np.load(path)["meta"]))


def test_side_chain_kind_accepts_complete_candidates_with_glycine(pipeline, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    s, t = task(pipeline, out, kind="sc_rel")
    name, err, status = s.process_entry(t)
    assert err is None, err
    z = np.load(out / "TEST_A.npz")
    assert len(read_meta(out / "TEST_A.npz")["candidates"]) == 6 and np.all(z["coverage_mask"] == 1.0)
    # the cause: glycine has NO side-chain value, so "finite everywhere" can never hold even for a perfect candidate
    mask = z["mask"].astype(bool)
    gly = np.array([c == "G" for c in SEQ])
    assert (gly & mask).any() and np.isnan(z["y"][gly & mask]).all()
    assert not np.isfinite(z["c"][0][gly & mask]).any() and np.isfinite(z["c"][0][~gly & mask]).all()


def test_failed_rerun_never_leaves_a_stale_entry_in_the_tables(pipeline, tmp_path):
    out = tmp_path / "o"
    base = [ROOT / "score_candidates.py", "--csv", pipeline["ds"], "--truth-dir", pipeline["truth"], "--cache-dir", pipeline["cache"],
            "--out", out, "--workers", 1]
    r = run(base + ["--candidates", str(pipeline["preds"] / "{entry}" / "*.cif.gz")])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TEST_A" in (out / "entry_summary.csv").read_text()
    r = run(base + ["--candidates", str(tmp_path / "nowhere" / "{entry}" / "*.cif.gz")])          # the entry now fails
    assert r.returncode == 1 and "FAILED TEST_A" in r.stdout
    assert not (out / "TEST_A.npz").exists() and list((out / "_quarantine").glob("TEST_A.*.npz"))
    assert not (out / "entry_summary.csv").exists() and not (out / "candidate_scores.csv").exists()   # no stale table left behind
    # results outside the current CSV selection are not collected either
    r = run(base + ["--candidates", str(pipeline["preds"] / "{entry}" / "*.cif.gz")])
    assert r.returncode == 0 and "TEST_A" in (out / "entry_summary.csv").read_text()
    other = tmp_path / "other.csv"
    other.write_text("pdb_id,chain_id,label_chain_id,sequence\n" f"ZZZZ,A,A,{SEQ}\n")
    r = run([ROOT / "score_candidates.py", "--csv", other, "--out", out, "--collect-only"])
    assert r.returncode == 0 and "collected 0 entries" in r.stdout and not (out / "entry_summary.csv").exists()


def make_preds_copy(pipeline, tmp_path, breaks=()):
    """Copy of the six candidates; the named ones are replaced by files of a DIFFERENT sequence (they will fail)."""
    full = pipeline["full"]
    d = tmp_path / "preds" / "TEST_A"
    d.mkdir(parents=True)
    other = "".join("A" if c != "A" else "G" for c in SEQ)
    wrong = AtomicStructure(full.coords, full.elements, full.atom_names, full.residue_index,
                            np.array([{"A": "ALA", "G": "GLY"}[c] for c in other]))
    for f in (pipeline["preds"] / "TEST_A").glob("*.cif.gz"):
        if f.name in breaks:
            write_chain_cif(d / f.name, wrong)
        else:
            (d / f.name).write_bytes(f.read_bytes())
    return str(tmp_path / "preds" / "{entry}" / "*.cif.gz")


def write_manifest(path, scores):
    path.write_text("entry,candidate,score\n" + "\n".join(f"TEST_A,{k}.cif.gz,{v}" for k, v in scores.items()) + "\n")
    return path


def test_original_ranks_are_kept_when_a_candidate_fails(pipeline, tmp_path):
    s = load_script("score_candidates")
    scores = {"c03": 0.99, "c00": 0.9, "c01": 0.8, "c02": 0.7, "c04": 0.6, "c05": 0.5}            # c03 is ranked FIRST
    # (a) the top-ranked candidate fails: no survivor may be promoted to "top 1", and top-5 contains the failure too
    out = tmp_path / "a"
    out.mkdir()
    pat = make_preds_copy(pipeline, tmp_path / "pa", breaks=("c03.cif.gz",))
    _, t = task(pipeline, out, pat)
    assert s.process_entry(t)[1] is None
    n, bad = s.collect(out, ranking=s.read_ranking(write_manifest(tmp_path / "ma.csv", scores)), entries={"TEST_A"})
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    assert e["rmsd_top1"] == "" and e["rmsd_best_top5"] == "" and e["n_manifest_candidates"] == "6" and e["n_manifest_not_scored"] == "1"
    assert "top1" in e["topN_unavailable_reason"] and e["ranking_status"].startswith("incomplete")
    ranks = {r["candidate"]: r["rank"] for r in csv.DictReader(open(out / "candidate_scores.csv"))}
    assert ranks["c00.cif.gz"] == "2" and ranks["c05.cif.gz"] == "6"                               # original ranks, with a gap at 1
    # (b) the LOWEST-ranked candidate fails: top 1 and top 5 are still exact, top 20 (= everything) is not
    out = tmp_path / "b"
    out.mkdir()
    pat = make_preds_copy(pipeline, tmp_path / "pb", breaks=("c05.cif.gz",))
    _, t = task(pipeline, out, pat)
    assert s.process_entry(t)[1] is None
    s.collect(out, ranking=s.read_ranking(write_manifest(tmp_path / "mb.csv", scores)), entries={"TEST_A"})
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    rm = {r["candidate"]: float(r["rmsd_ca"]) for r in csv.DictReader(open(out / "candidate_scores.csv"))}
    assert float(e["rmsd_top1"]) == pytest.approx(rm["c03.cif.gz"])
    assert float(e["rmsd_best_top5"]) == pytest.approx(min(rm[f"c0{i}.cif.gz"] for i in (3, 0, 1, 2, 4)))
    assert e["rmsd_best_top20"] == ""


def test_reuse_rechecks_min_candidates(pipeline, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    s, t = task(pipeline, out)
    assert s.process_entry(t)[2] == "scored"
    assert s.process_entry(t)[2] == "reused"
    s, t20 = task(pipeline, out, min_candidates=20)
    name, err, status = s.process_entry(t20)
    assert status == "failed" and "only 6 usable candidate" in err and "20 required" in err
    assert not (out / "TEST_A.npz").exists() and list((out / "_quarantine").glob("TEST_A.*.npz"))


def test_ranking_rows_are_never_merged_or_overwritten():
    s = load_script("score_candidates")
    meta = {"entry": "TEST_A", "pdb_id": "TEST", "candidates": ["s1/m.cif.gz", "s2/m.cif.gz", "s1/other.cif.gz"]}

    def rk(rows):
        return s.Ranking([{"entry": e, "cand": s._norm(c), "score": float(v)} for e, c, v in rows])

    # two different candidates with the same file name: a bare name is AMBIGUOUS, never silently assigned
    r = s.resolve_ranking(rk([("TEST_A", "m.cif.gz", 1.0), ("TEST_A", "other.cif.gz", 3.0)]), meta)
    assert any("ambiguous" in p for p in r["problems"]) and np.isnan(r["scores"][0]) and np.isnan(r["scores"][1]) and r["scores"][2] == 3.0
    # path-qualified rows (relative, absolute, Windows separators) identify each candidate exactly
    r = s.resolve_ranking(rk([("TEST_A", "s1/m.cif.gz", 1.0), ("TEST", "C:\\runs\\TEST_A\\s2\\m.cif.gz", 2.0), ("TEST_A", "s1/other", 3.0)]), meta)
    assert not r["problems"] and list(r["scores"]) == [1.0, 2.0, 3.0]
    # the same candidate with two different scores is a conflict; the same score twice is just a duplicate row
    r = s.resolve_ranking(rk([("TEST_A", "s1/m.cif.gz", 1.0), ("TEST_A", "s1/m.cif.gz", 5.0)]), meta)
    assert any("conflict" in p for p in r["problems"]) and r["scores"][0] == 1.0
    r = s.resolve_ranking(rk([("TEST_A", "s1/m.cif.gz", 1.0), ("TEST_A", "s1/m.cif.gz", 1.0)]), meta)
    assert not r["problems"]
    # rows for other entries are ignored; a row matching nothing stays in the manifest as an unscored (failed/missing) candidate
    r = s.resolve_ranking(rk([("OTHER_A", "s1/m.cif.gz", 9.0), ("TEST_A", "gone.cif.gz", 2.5)]), meta)
    assert [m for m in r["manifest"]] == [("gone.cif.gz", 2.5, None)]
    # top-N rule: an unscored candidate invalidates exactly those N it could belong to
    rows = [("TEST_A", "s1/m.cif.gz", 1.0), ("TEST_A", "s2/m.cif.gz", 2.0), ("TEST_A", "s1/other.cif.gz", 3.0)]
    mid = s.resolve_ranking(rk(rows + [("TEST_A", "gone.cif.gz", 2.5)]), meta)          # failed candidate ranked 2nd of 4
    assert s.top_n_status(mid, 3, 1)[0] and not s.top_n_status(mid, 3, 2)[0] and not s.top_n_status(mid, 3, 3)[0]
    low = s.resolve_ranking(rk(rows + [("TEST_A", "gone.cif.gz", 0.1)]), meta)          # failed candidate ranked last of 4
    assert s.top_n_status(low, 3, 1)[0] and s.top_n_status(low, 3, 3)[0] and not s.top_n_status(low, 3, 4)[0]
    tie = s.resolve_ranking(rk(rows + [("TEST_A", "gone.cif.gz", 3.0)]), meta)          # a tie with the top score is not safe either
    assert not s.top_n_status(tie, 3, 1)[0]
    assert s.top_n_status(s.resolve_ranking(rk(rows), meta), 3, 20)[0]                  # nothing missing: any N is fine
    part = s.resolve_ranking(rk(rows[:2]), meta)                                         # a scored candidate has no score at all
    assert not s.top_n_status(part, 3, 1)[0]


def test_same_file_names_in_different_folders_score_and_rank_separately(pipeline, tmp_path):
    s = load_script("score_candidates")
    root = tmp_path / "preds" / "TEST_A"
    for sub, src in (("s1", "c00.cif.gz"), ("s2", "c04.cif.gz")):
        (root / sub).mkdir(parents=True)
        (root / sub / "m_model.cif.gz").write_bytes((pipeline["preds"] / "TEST_A" / src).read_bytes())
    out = tmp_path / "o"
    out.mkdir()
    _, t = task(pipeline, out, str(tmp_path / "preds" / "{entry}" / "*" / "*.cif.gz"))
    assert s.process_entry(t)[1] is None
    assert read_meta(out / "TEST_A.npz")["candidates"] == ["s1/m_model.cif.gz", "s2/m_model.cif.gz"]
    bare = tmp_path / "bare.csv"
    bare.write_text("entry,candidate,score\nTEST_A,m_model.cif.gz,1\nTEST_A,m_model.cif.gz,2\n")      # same name twice: cannot be told apart
    n, bad = s.collect(out, ranking=s.read_ranking(bare), entries={"TEST_A"})
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    assert bad == 1 and e["ranking_status"].startswith("conflict: ambiguous") and e["rmsd_top1"] == ""
    good = tmp_path / "good.csv"
    good.write_text("entry,candidate,score\nTEST_A,s1/m_model.cif.gz,1\nTEST_A,s2/m_model.cif.gz,2\n")
    n, bad = s.collect(out, ranking=s.read_ranking(good), entries={"TEST_A"})
    e = list(csv.DictReader(open(out / "entry_summary.csv")))[0]
    rm = {r["candidate"]: float(r["rmsd_ca"]) for r in csv.DictReader(open(out / "candidate_scores.csv"))}
    assert bad == 0 and e["ranking_status"] == "complete" and float(e["rmsd_top1"]) == pytest.approx(rm["s2/m_model.cif.gz"])
    r = run([ROOT / "score_candidates.py", "--csv", pipeline["ds"], "--out", out, "--collect-only", "--ranking-csv", bare])
    assert r.returncode == 1                                                                         # a conflicting manifest fails loudly


def test_truth_and_cache_must_be_the_same_structure(pipeline, tmp_path):
    truth, full = pipeline["truth"], pipeline["full"]
    assert read_meta(truth / "TEST_A.npz")["source_cif_sha1"]
    # a cache whose COORDINATES changed but whose residue coverage did not (the old resolved-mask check would accept it)
    cache2 = tmp_path / "cache2"
    cache2.mkdir()
    s_true = load_chain_from_mmcif(pipeline["cache"] / "TEST.cif.gz", "A", SEQ)[0]
    ca, has = ca_coordinates(s_true)
    write_chain_cif(cache2 / "TEST.cif.gz", s_true, smooth_displacement(s_true.coords, ca[has], 1.5, seed=5))
    out = tmp_path / "o"
    out.mkdir()
    s, t = task(pipeline, out, cache=cache2)
    name, err, status = s.process_entry(t)
    assert status == "failed" and "content hash differs" in err
    # a truth file from an older script (no hash): accepted with a note; --verify-truth then really recomputes and compares
    old_truth = tmp_path / "old_truth"
    old_truth.mkdir()
    with np.load(truth / "TEST_A.npz") as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    meta.pop("source_cif_sha1")
    np.savez_compressed(old_truth / "TEST_A.npz", meta=np.array(json.dumps(meta)), **arrays)
    out2 = tmp_path / "o2"
    out2.mkdir()
    s, t = task(pipeline, out2, truth=old_truth, cache=cache2)
    assert s.process_entry(t)[1] is None and "no source-structure hash" in read_meta(out2 / "TEST_A.npz")["truth_check"]
    out3 = tmp_path / "o3"
    out3.mkdir()
    s, t = task(pipeline, out3, truth=old_truth, cache=cache2, verify_truth=True)
    name, err, status = s.process_entry(t)
    assert status == "failed" and "verify-truth" in err
    out4 = tmp_path / "o4"
    out4.mkdir()
    s, t = task(pipeline, out4, truth=old_truth, verify_truth=True)                                  # the true cache passes the check
    assert s.process_entry(t)[1] is None and "matched" in read_meta(out4 / "TEST_A.npz")["truth_check"]


def test_truth_residue_table_exports_side_chain_columns(pipeline):
    import gzip as gz
    with gz.open(pipeline["truth"] / "residues.csv.gz", "rt") as fh:
        rows = list(csv.DictReader(fh))
    assert {"sc_abs_1.4", "sc_rel_1.4", "sc_abs_6", "sc_rel_6", "abs_1.4", "rel_6"} <= set(rows[0])
    gly = [r for r in rows if r["residue"] == "GLY" and r["in_mask"] == "1"]
    ala = [r for r in rows if r["residue"] == "LYS" and r["in_mask"] == "1"]
    assert gly and all(r["sc_rel_1.4"] == "" and r["rel_1.4"] != "" for r in gly)                   # Gly: no side chain, but a whole-residue value
    assert ala and all(r["sc_rel_1.4"] != "" for r in ala)
