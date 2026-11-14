"""
Download the still-available photos of the SHAPY Model Agency Data.

The dataset (dataset/shapy/ModelAgencyData, from shapy.is.tue.mpg.de) gives,
for ~4,700 fashion models, height / bust / waist / hips and links to photos
on their agency websites. Many links have died since 2022; this script
keeps only models whose photos still load.

Polite: few parallel requests per website and a short pause between them.
Licence: SHAPY data is for non-commercial scientific research only.

Output: dataset/shapy/model_agency_images/<agency>/<model_id>/<n>.jpg
        dataset/shapy/model_agency_images/index.csv  (one row per model)

  python scripts/download_model_agency.py            # resumes if interrupted
"""

import os
import re
import csv
import json
import time
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
SRC = data_path('shapy', 'ModelAgencyData')
OUT = data_path('shapy', 'model_agency_images')
MAX_PER_MODEL = 6
WORKERS_PER_SITE = 3
socket.setdefaulttimeout(10)


def fetch(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research; TailorCam)'})
        with urllib.request.urlopen(req) as r:
            if r.status == 200 and r.headers.get_content_type().startswith('image'):
                return r.read()
    except Exception:
        pass
    return None


def model_id(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')[:60] or 'model'


def do_model(job):
    agency, name, urls, meta = job
    folder = os.path.join(OUT, agency, model_id(name))
    have = sorted(f for f in os.listdir(folder)) if os.path.isdir(folder) else []
    if have:
        return {**meta, 'n_images': len(have)}
    first = fetch(urls[0])                      # dead first link → skip the model
    if first is None:
        return None
    os.makedirs(folder, exist_ok=True)
    saved = 0
    for k, u in enumerate(urls[:MAX_PER_MODEL]):
        data = first if k == 0 else fetch(u)
        if data and len(data) > 5000:
            with open(os.path.join(folder, f'{k:02d}.jpg'), 'wb') as f:
                f.write(data)
            saved += 1
        time.sleep(0.2)
    return {**meta, 'n_images': saved} if saved else None


def main():
    data = json.load(open(os.path.join(SRC, 'cleaned_model_data.json')))
    split = json.load(open(os.path.join(SRC, 'weight_splits.json')))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for agency, d in data.items():
        jobs = []
        for i, name in enumerate(d['model_name']):
            if not d['has_relevant_vars'][i] or not d['image_urls'][i]:
                continue
            meta = {'agency': agency, 'model_name': name, 'model_id': model_id(name),
                    'split': split.get(agency, {}).get(name, ''), 'gender': d['gender'][i],
                    'height_cm': d['height_cm'][i], 'chest_cm': d['bust_cm'][i],
                    'waist_cm': d['waist_cm'][i], 'hip_cm': d['hips_cm'][i]}
            jobs.append((agency, name, d['image_urls'][i], meta))
        t0 = time.time()
        with ThreadPoolExecutor(WORKERS_PER_SITE) as ex:
            got = [r for r in ex.map(do_model, jobs) if r]
        rows += got
        print(f'{agency:16s} {len(got):4d}/{len(jobs)} models downloaded '
              f'({sum(r["n_images"] for r in got)} photos, {time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(OUT, 'index.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\nTOTAL {len(rows)} models, {sum(r["n_images"] for r in rows)} photos -> {OUT}')


if __name__ == '__main__':
    main()
