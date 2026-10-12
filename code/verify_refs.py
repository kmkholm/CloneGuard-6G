"""Verify every DOI of paper/refs.bib on Crossref and rewrite the entry with the registered metadata
(authors, title, container, volume, number, pages, year). Entries without DOI are kept and listed.
python verify_refs.py  -> paper/refs.bib rewritten (backup refs_before_check.bib), paper/refs_check.txt"""
import os, re, time, shutil, requests
HERE = os.path.dirname(os.path.abspath(__file__)); BIB = os.path.join(HERE, "..", "paper", "refs.bib")
txt = open(BIB, encoding="utf-8").read()


def split_entries(t):
    out, i = [], 0
    while True:
        i = t.find("@", i)
        if i < 0:
            break
        j = t.find("{", i); typ = t[i + 1:j].strip(); k = t.find(",", j); key = t[j + 1:k].strip()
        depth, p = 1, j + 1
        while depth and p < len(t):
            depth += {"{": 1, "}": -1}.get(t[p], 0); p += 1
        out.append((typ, key, t[k + 1:p - 1])); i = p
    return out


def field(body, k):
    m = re.search(k + r"\s*=\s*\{", body)
    if not m:
        return None
    depth, p = 1, m.end()
    while depth and p < len(body):
        depth += {"{": 1, "}": -1}.get(body[p], 0); p += 1
    return body[m.end():p - 1].strip()


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%")


entries = split_entries(txt)
assert entries, "no entries parsed - refs.bib untouched"
shutil.copy(BIB, os.path.join(HERE, "..", "paper", "refs_before_check.bib"))
out, rep = [], []
for typ, key, body in entries:
    doi = field(body, "doi")
    if not doi:
        rep.append(f"NO-DOI  {key}: kept as written (check manually)")
        out.append(f"@{typ}{{{key},{body}}}"); continue
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30, headers={"User-Agent": "refcheck/1.0 (mailto:kmkhol01@gmail.com)"})
    if r.status_code != 200:
        rep.append(f"FAIL    {key}: {doi} -> HTTP {r.status_code}"); out.append(f"@{typ}{{{key},{body}}}"); continue
    m = r.json()["message"]
    auth = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", []))
    title = esc(re.sub(r"<[^>]+>", "", m.get("title", [""])[0]))
    cont = esc((m.get("container-title") or [""])[0])
    year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    vol, num, pages = m.get("volume"), m.get("issue"), m.get("page")
    if m.get("type") == "journal-article":
        e = f"@article{{{key},\n  author={{{auth}}},\n  title={{{title}}},\n  journal={{{cont}}}, year={{{year}}}"
        if vol: e += f", volume={{{vol}}}"
        if num: e += f", number={{{num}}}"
        if pages: e += f", pages={{{pages.replace('-', '--')}}}"
    else:
        e = f"@inproceedings{{{key},\n  author={{{auth}}},\n  title={{{title}}},\n  booktitle={{{cont}}}, year={{{year}}}"
        if pages: e += f", pages={{{pages.replace('-', '--')}}}"
    e += f", doi={{{doi}}}}}"
    out.append(e)
    rep.append(f"OK      {key}: {year} {title[:70]} | {cont[:40]} | {auth[:60]}")
    time.sleep(0.3)
open(BIB, "w", encoding="utf-8").write("\n".join(out) + "\n")
open(os.path.join(HERE, "..", "paper", "refs_check.txt"), "w", encoding="utf-8").write("\n".join(rep))
print("\n".join(rep))
