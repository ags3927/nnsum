import argparse
import pathlib
import ujson as json
import json as original_json

import torch
import nnsum
import pandas as pd
import rouge_papier
from multiprocessing import cpu_count

from typing import *
from rouge import Rouge
rouge = Rouge()


def eval_rouge_bangla(can: str, ref: str) -> Tuple[float, float, float]:
    """
        Use Rouge library to naively calculate rouge score for Bangla
        :param can: Hypothesis/candidate summary
        :param ref: Ground Truth summary
        :return:
    """
    if can == '' or ref == '':
        return 0.0, 0.0, 0.0
    scores = rouge.get_scores(can, ref)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']


def main():
    parser = argparse.ArgumentParser(
        "Evaluate nnsum models using original Perl ROUGE script.")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--sentence-limit", default=None, type=int)
    parser.add_argument("--summary-length", type=int, default=100)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--inputs", type=pathlib.Path, required=True)
    parser.add_argument(
        "--refs", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--results", type=pathlib.Path, required=False, default=None)
 
    args = parser.parse_args() 

    if args.loader_workers is None:
        args.loader_workers = min(16, cpu_count())

    print("Loading model...", end="", flush=True)
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    # model = torch.load(args.model)
    if args.gpu > -1:
        model.cuda(args.gpu)
    vocab = model.embeddings.vocab
    print(" OK!")

    data = nnsum.data.SummarizationDataset(
        vocab,
        args.inputs,
        references_dir=args.refs,
        sentence_limit=args.sentence_limit)
    loader = nnsum.data.SummarizationDataLoader(
        data, batch_size=args.batch_size, num_workers=args.loader_workers)

    gen_summary_path = '/home/ags/academics/thesis/nnsum/output/summaries'
    ids = []
    path_data = []
    model.eval()
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    with rouge_papier.util.TempFileManager() as manager:
        with torch.no_grad():
            for step, batch in enumerate(loader, 1):
                batch = batch.to(args.gpu)
                print("generating summaries {} / {} ...".format(
                        step, len(loader)),
                    end="\r" if step < len(loader) else "\n", flush=True)
                texts, sent_ids, doc_ids = model.predict(batch, return_indices=True, max_length=args.summary_length)
                
                
                sorted_texts = []
                
                for text_arr, sent_id_arr in zip(texts, sent_ids):
                    summary = []
                    for sent, sent_id in zip(text_arr, sent_id_arr):
                        summary.append({
                            "sentence": sent,
                            "sentence_id": sent_id
                        })
                    summary.sort(key=lambda x: x["sentence_id"])    
                    sorted_texts.append(summary)
                
                for idx, doc_id in enumerate(doc_ids):
                    with open(gen_summary_path + '/' + str(doc_id) + '.json', 'w', encoding='utf8') as f:
                        summary_sentences = [sorted_texts[idx][sentence_idx]["sentence"] for sentence_idx in range(len(sorted_texts[idx]))]
                        summary_text = "\n".join(summary_sentences)
                        f1 = open(str(args.refs) + '/' + str(doc_id) + ".spl", 'r')
                        ref_text = f1.read()
                        rouge_1, rouge_2, rouge_l = eval_rouge_bangla(summary_text, ref_text)
                        
                        summary_obj = {
                            'id': doc_id,
                            'summary': summary_text,
                            'ground_truth': ref_text,
                            'rouge-1': rouge_1,
                            'rouge-2': rouge_2,
                            'rouge-l': rouge_l
                        }
                        rouge_1_scores.append(rouge_1)
                        rouge_2_scores.append(rouge_2)
                        rouge_l_scores.append(rouge_l)
                        
                        original_json.dump(summary_obj, f, ensure_ascii=False)
                        
                
                # for idx, doc_id in enumerate(doc_ids):
                #     summary_obj = {
                #             'id': doc_id,
                #             'summary': sorted_texts[idx]
                #     }
                #     original_json.dump(summary_obj, f, ensure_ascii=False)
                #     summary_sentences = [sorted_text[idx][sentence_idx]["sentence"] for sentence_idx in range(len(sorted_text[idx]))]
                #     summary_text = "\n".join(summary_sentences)
                #     f = open(args.refs + doc_id + ".spl")
                #     ref_text = f.read()
                #     rouge_1, rouge_2, rouge_l = eval_rouge_bangla(summary_text, ref_text)
                
                
                    
                for text, ref_paths in zip(texts, batch.reference_paths):
                    summary = "\n".join(text)                
                    summary_path = manager.create_temp_file(summary)
                    path_data.append(
                        [summary_path, [str(x) for x in ref_paths]])
                ids.extend(batch.id)

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=True, 
            remove_stopwords=args.remove_stopwords,
            length=args.summary_length)
        df.index = ids + ["average"]
        df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
        print(df[-1:])
       
        if args.results:
            records = df[:-1].to_dict("records")

            results = {"idividual": {id: record 
                                     for id, record in zip(ids, records)},
                       "average": df[-1:].to_dict("records")[0]}
            args.results.parent.mkdir(parents=True, exist_ok=True)
            # with args.results.open("w") as fp:
            #     fp.write(json.dumps(results))

        num_of_test_data = len(loader)*args.batch_size
        avg_rouge_1 = (sum(rouge_1_scores)/num_of_test_data)
        avg_rouge_2 = (sum(rouge_2_scores)/num_of_test_data)
        avg_rouge_l = (sum(rouge_l_scores)/num_of_test_data)
        results = {
            "Rouge-1": avg_rouge_1,
            "Rouge-2": avg_rouge_2,
            "Rouge-L": avg_rouge_l
        }
        with args.results.open("w") as fp:
            fp.write(json.dumps(results))
        
        print("Average rouge-1 score = " + str(avg_rouge_1) + "\n")
        print("Average rouge-2 score = " + str(avg_rouge_2) + "\n")
        print("Average rouge-l score = " + str(avg_rouge_l) + "\n")
        
if __name__ == "__main__":
    main()
