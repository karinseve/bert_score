import time
import os
import sys
import torch
import argparse
import pandas as pd

from bert_score.utils import (get_model, get_idf_dict, bert_cos_score_idf,
                    get_bert_embedding, model_types,
                    lang2model, model2layers, get_hash,
                    cache_scibert, sent_encode)

from collections import defaultdict
from transformers import AutoTokenizer


class BertScore:

    def __init__(self, args):

        self.rescale_with_baseline = args.rescale_with_baseline
        self.model_type = args.model
        self.lang = args.lang
        self.num_layers = args.num_layers
        self.verbose = args.verbose
        self.batch_size = args.batch_size
        self.nthreads = args.nthreads
        self.all_layers = None
        self.return_hash = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.idf_dict = None
        self.tokenizer = None
        self.model = None
        
        self.load()

    def main(self):

        # interactive session
        print("Enter reference sentence starting with an 'r:' and the candidate sentence "
              "starting with a 'c:'.\nIf you don't specify any of these again it will keep "
              "using the same.\n\nEnter q to stop\n")

        ref = ''
        cand = ''
        while True:
            query = input("> ")
            if query == 'q':
                break
            else:
                toks = query.split(":")
                if len(toks) == 1 or not (toks[0] == 'r' or toks[0] == 'c'):
                    print("You must enter 'r:' or 'c:' first!")
                if toks[0] == "r":
                    ref = toks[1].strip()
                elif toks[0] == "c":
                    cand = toks[1].strip()

            if not (ref == '' or cand== ''):
                (P, R, F) = self.score(ref, cand)
                print(f'P={P.mean():.6f} R={R.mean():.6f} F={F.mean():.6f}')

    def load(self):

        assert self.lang is not None or self.model_type is not None, \
            'Either lang or model_type should be specified'

        if self.rescale_with_baseline:
            assert self.lang is not None, 'Need to specify Language when rescaling with baseline'

        if self.model_type is None:
            self.lang = self.lang.lower()
            self.model_type = lang2model[self.lang]
        if self.num_layers is None:
            self.num_layers = model2layers[self.model_type]

        if self.model_type.startswith('scibert'):
            self.tokenizer = AutoTokenizer.from_pretrained(cache_scibert(self.model_type))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.model = get_model(self.model_type, self.num_layers, self.all_layers)
        self.model.to(self.device)

        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0

    def score(self, ref, cand):
        return self.score_individual(self.model, ref, cand, self.tokenizer, self.idf_dict,
                                       verbose=self.verbose, device=self.device,
                                       batch_size=self.batch_size, all_layers=self.all_layers,
                                       rescale_with_baseline=self.rescale_with_baseline, return_hash=self.return_hash,
                                       lang=self.lang, model_type=self.model_type, num_layers=self.num_layers)


    def score_individual(self, model, ref, cand, tokenizer, idf_dict,
                                   verbose=False, device='cpu',
                                   batch_size=64, all_layers=None,
                                   rescale_with_baseline=None, return_hash=None,
                                   lang='en', model_type=None, num_layers=None):
        start = time.perf_counter()
        all_preds = bert_cos_score_idf(model, [ref], [cand], tokenizer, idf_dict,
                                       verbose=verbose, device=device,
                                       batch_size=batch_size, all_layers=all_layers).cpu()

        if rescale_with_baseline:
            baseline_path = os.path.join(
                os.path.dirname(__file__),
                f'rescale_baseline/{lang}/{model_type}.tsv'
            )
            if os.path.isfile(baseline_path):
                if not all_layers:
                    baselines = torch.from_numpy(
                        pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                    )[1:].float()
                else:
                    baselines = torch.from_numpy(
                        pd.read_csv(baseline_path).to_numpy()
                    )[:, 1:].unsqueeze(1).float()

                all_preds = (all_preds - baselines) / (1 - baselines)
            else:
                print(f'Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}', file=sys.stderr)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2] # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(ref) / time_diff:.2f} sentences/sec')

        if return_hash:
            return tuple([out, get_hash(model_type, num_layers, False, rescale_with_baseline)])

        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Calculate BERTScore - Interactive Session')
    parser.add_argument('--lang', type=str, default=None,
                        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text')
    parser.add_argument('--rescale_with_baseline', action='store_true',
                        help='Rescaling the numerical score with precomputed baselines')
    parser.add_argument('-m', '--model', default=None,
                        help='BERT model name (default: bert-base-uncased) or path to a pretrain model')
    parser.add_argument('-l', '--num_layers', type=int, default=None, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--nthreads', type=int, default=4, help='number of cpu workers (default: 4)')
    parser.add_argument('--idf', action='store_true', help='BERT Score with IDF scaling')
    parser.add_argument('--rescale-with-baseline', action='store_true',
                        help='Rescaling the numerical score with precomputed baselines')
    parser.add_argument('-s', '--seg_level', action='store_true', help='show individual score of each pair')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    scorer = BertScore(args)
    scorer.main()
