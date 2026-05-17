import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.noise import generate_noisy_variants


def main(input_path: str, output_path: str, variants: int):
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open('r', encoding='utf-8') as fin, out.open('w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc='Creating pairs'):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text') or obj.get('review') or obj.get('sentence')
            if not text:
                continue

            # generate variants
            variants_list = generate_noisy_variants(text, variants)
            for v in variants_list:
                pair = {'noisy': v, 'clean': text}
                fout.write(json.dumps(pair, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/absa_training_flat.jsonl')
    parser.add_argument('--output', '-o', default='data/denoise_pairs.jsonl')
    parser.add_argument('--variants', '-n', type=int, default=2)
    args = parser.parse_args()
    main(args.input, args.output, args.variants)
