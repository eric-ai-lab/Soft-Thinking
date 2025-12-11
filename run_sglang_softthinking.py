import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, AIMEEvaluator
import asyncio
import matheval
import humanevaleval
import mbppeval
from huggingface_hub import HfApi
import torch
import time
import convert_livecodebench

MATH_DATASETS = ["math500","aime2024","aime2025","gpqa_diamond","gsm8k","amc23"]
CODE_DATASETS = ["humaneval","mbpp","livecodebench"]

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')
    parser.add_argument('--dataset', type=str, choices=["math500", "aime2024", "aime2025", "gpqa_diamond", "gsm8k", "amc23", "humaneval", "mbpp", "livecodebench"], help='Name of dataset')
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer", help='Sampling backend')
    parser.add_argument('--model_name', type=str, required=True, default="Qwen/QwQ-32B", help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tensor parallel size, tp_size)')
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None, help='Max number of batch runned in one time.')
    parser.add_argument('--max_running_requests', type=int, default=None, help='Max number of requests runned together.')
    parser.add_argument('--max_batch', type=int, default=1000000, help='Max number of batch runned in one time.')
    parser.add_argument('--mem_fraction_static', type=float, default=0.5, help='Max memory to use per gpu.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--end_idx', type=int, default=500, help='End index for processing samples')

    # sampling parameters
    parser.add_argument('--num_samples', type=int, default=1, help='Sampling number')
    parser.add_argument('--max_generated_tokens', type=int, default=32768, help='Limit the number of generated tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling probability')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling probability')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min-p sampling probability')
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6, help='Temperature after thinking')
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95, help='Top-p after thinking')
    parser.add_argument('--after_thinking_top_k', type=int, default=30, help='Top-k after thinking')
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0, help='Min-p after thinking')
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0, help='Early stopping entropy threshold (set it to 0.0 to disable early stopping)')
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256, help='Early stopping length threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')

    # Noise parameters
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0, help='Dirichlet alpha')
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=1.0, help='Gumbel-softmax temperature')
    parser.add_argument('--add_noise_dirichlet', action='store_true', help='Add Dirichlet noise to sampling')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true', help='Add Gumbel-softmax noise to sampling')

    # Eval & Push parameters
    parser.add_argument('--reeval', action='store_true', help='Enable re-evaluation for code datasets (due to Multiprocessing bug when using sglang rollout)')
    parser.add_argument('--use_llm_judge', action='store_true', help='Enable LLM judge')
    parser.add_argument('--api_base', type=str, default=None, help='')
    parser.add_argument('--deployment_name', type=str, default=None, help='')
    parser.add_argument('--api_version', type=str, default=None, help='')
    parser.add_argument('--api_key', type=str, default=None, help='')
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14", help='Judge LLM model name for evaluation')
    parser.add_argument('--push_results_to_hf', action='store_true', help='Enable push to huggingface')
    parser.add_argument('--hf_token', type=str, default=None, help='')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='')

    parser.add_argument("--enable_soft_thinking", action="store_true", help="Enable soft thinking mode")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=15)

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    max_generated_tokens = args.max_generated_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    think_end_str = args.think_end_str
    # num_samples = args.num_samples
    random_seed = args.random_seed
    num_gpus = args.num_gpus
    max_running_requests = args.max_running_requests
    max_batch = args.max_batch
    mem_fraction_static = args.mem_fraction_static
    start_idx = args.start_idx
    end_idx = args.end_idx
    reeval = args.reeval


    print(f"Arguments: {args}", flush=True)
    if dataset in MATH_DATASETS:
        matheval.set_client(args.api_base, args.deployment_name, args.api_version, args.api_key, args.judge_model_name)

    # load dataset
    if dataset == "math500":
        with open("./datasets/math500.json") as f:
            samples = json.load(f)
    elif dataset == "aime2024":
        with open("./datasets/aime2024.json") as f:
            samples = json.load(f)
    elif dataset == "aime2025":
        with open("./datasets/aime2025.json") as f:
            samples = json.load(f)
    elif dataset == "gpqa_diamond":
        with open("./datasets/gpqa_diamond.json") as f:
            samples = json.load(f)
    elif dataset == "gsm8k":
        with open("./datasets/gsm8k.json") as f:
            samples = json.load(f)
    elif dataset == "amc23":
        with open("./datasets/amc23.json") as f:
            samples = json.load(f)
    elif dataset == "humaneval":
        with open("./datasets/humaneval.json") as f:
            samples = json.load(f)
    elif dataset == "mbpp":
        with open("./datasets/mbpp.json") as f:
            samples = json.load(f)
    elif dataset == "livecodebench":
        with open("./datasets/livecodebench.json") as f:
            samples = json.load(f)
    else:
        raise ValueError("Invalid dataset name")

    # prompt templates
    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

    GPQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,"answer": "C".

{Question}
""".strip()
    
    CODE_QUERY_TEMPLATE = """
Please solve the programming task below in Python. Code should be wrapped in a markdown code block.

```python
{Question}
```
""".strip()

    MBPP_QUERY_TEMPLATE = """
Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:
1. The function name and signature must match exactly as specified in the test cases.
2. Your code should be wrapped in a markdown code block without including any test cases.

Task:
{Question}

Test Cases:
```python
{TestCases}
```
""".strip()
    def get_lcb_prompt(question_content, starter_code):
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"Question: {question_content}\n\n"
        if starter_code:
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return prompt


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p, "repetition_penalty": args.repetition_penalty,
                        "after_thinking_temperature": args.after_thinking_temperature, "after_thinking_top_p": args.after_thinking_top_p, "after_thinking_top_k": args.after_thinking_top_k, "after_thinking_min_p": args.after_thinking_min_p,
                        "n": 1, # repeat prompt for num_samples times instead of using num_samples in sampling_params
                        "gumbel_softmax_temperature": args.gumbel_softmax_temperature, "dirichlet_alpha": args.dirichlet_alpha,
                        "max_new_tokens": max_generated_tokens, "think_end_str": think_end_str,
                        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
                        "early_stopping_length_threshold": args.early_stopping_length_threshold
                    }

    os.makedirs(f"{args.output_dir}/results/{dataset}", exist_ok=True)
    noise_suffix = (
        (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
        + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    base_filename = (
        f"{model_name.split('/')[-1]}_{dataset}_{args.enable_soft_thinking}_{args.num_samples}_"
        f"{temperature}_{top_p}_{top_k}_{min_p}_{args.repetition_penalty}_{args.dirichlet_alpha}_"
        f"{args.max_topk}_{max_generated_tokens}_{args.early_stopping_entropy_threshold}_"
        f"{args.early_stopping_length_threshold}{noise_suffix}"
    )
    results_file = f"{args.output_dir}/results/{dataset}/{base_filename}.json"
    results_statistics_file = f"{args.output_dir}/results/{dataset}/{base_filename}_statistics.json"

    results = []

    print("begin")
    start_time = time.time()

    # if reeval for code datasets, read results_file
    if reeval:
        # read results_file
        with open(results_file, "r") as f:
            results = json.load(f)
        prompt_list = []
        idx_list = list(range(start_idx, min(end_idx,len(results))))
        decoded_text_list = []
        finish_generation_list = []
        generated_tokens_list = []
        for r in results:
            prompt_list.append(r["prompt"])
            decoded_text_list.extend(r["completion"])
            finish_generation_list.extend(r["finish_generation"])
            generated_tokens_list.extend(r["generated_tokens"])
        results = []
    # if not reeval, collect prompt and idx
    else:
        prompt_list = []
        idx_list = []
        for idx in range(start_idx, min(end_idx,len(samples))):
            sample = samples[idx]

            if dataset in ["aime2024", "aime2025", "math500", "gsm8k", "amc23"]:
                chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "gpqa_diamond":
                chat = [{"role": "user", "content": GPQA_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "humaneval":
                chat = [{"role": "user", "content": CODE_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "mbpp":
                chat = [{"role": "user", "content": MBPP_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"], TestCases="\n".join(sample["final_answer"]["test_list"]))}]
            elif dataset == "livecodebench":
                chat = [{"role": "user", "content": get_lcb_prompt(question_content=sample["prompt"][0]["value"], starter_code=sample["final_answer"]["starter_code"])}]
            else:
                raise ValueError("Invalid dataset name")

            prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

            # Repeat prompt for num_samples times instead of using num_samples in sampling_params
            for _ in range(args.num_samples):
                prompt_list.append(prompt)

            idx_list.append(idx)

        # generate results
        decoded_text_list = []
        finish_generation_list = []
        generated_tokens_list = []
        idx = 0
        while idx < len(prompt_list):
            print(f"Number of GPUs available: {num_gpus}", flush=True)
            llm = sgl.Engine(
                model_path=model_name,
                tp_size=num_gpus,
                log_level="info",
                trust_remote_code=True,
                random_seed=random_seed,
                max_running_requests=max_running_requests,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=True,
                disable_overlap_schedule=True,
                enable_soft_thinking=args.enable_soft_thinking,
                add_noise_dirichlet=args.add_noise_dirichlet,
                add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
                max_topk=args.max_topk,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
                sampling_backend=args.sampling_backend
            )
            outputs =  llm.generate(prompt_list[idx:idx+max_batch], sampling_params)
            decoded_text_list.extend([o["text"] for o in outputs])
            finish_generation_list.extend([o["meta_info"]["finish_reason"]["type"] == "stop" and not args.enable_soft_thinking for o in outputs])

            generated_tokens_list.extend([o["meta_info"]["completion_tokens"] for o in outputs])
            idx += max_batch
            outputs = None
            llm.shutdown()

            torch.cuda.empty_cache()

    # init evaluator for code datasets
    if dataset in CODE_DATASETS:
        mbppeval.init_evaluator()
        humanevaleval.init_evaluator()

    # evaluate results
    for i,idx in enumerate(idx_list):
        print(idx, flush=True)
        sample = samples[idx]
        judge_info = []
        passat1_list = []
        decoded_text = decoded_text_list[i*args.num_samples:(i+1)*args.num_samples]
        finish_generation = finish_generation_list[i*args.num_samples:(i+1)*args.num_samples]

        # evaluate each sample
        for j in range(args.num_samples):
            for _ in range(5):
                try:
                    if dataset in MATH_DATASETS:
                        rule_judge_result = None
                        rule_judge_result, extracted_answer = matheval.evaluator_map[dataset].rule_judge(decoded_text[j],sample["final_answer"], finish_generation[j])
                        llm_judge_result = None
                        if not rule_judge_result and args.use_llm_judge:
                            llm_judge_result = matheval.evaluator_map[dataset].llm_judge(decoded_text[j],sample["final_answer"],extracted_answer, finish_generation[j])
                        finally_judge_result = rule_judge_result or llm_judge_result
                        judge_info.append({
                            "rule_judge_result": rule_judge_result,
                            "llm_judge_result": llm_judge_result,
                            "finally_judge_result": finally_judge_result
                        })
                        passat1_list.append(1.0 if finally_judge_result else 0.0)
                        # passat1 = sum(passat1_list)/len(passat1_list)

                    elif dataset in CODE_DATASETS:
                        k = 1
                        if dataset=="humaneval":
                            if reeval:
                                passat1, single_judge_info = humanevaleval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text[j],  sample["final_answer"], k)
                            else:
                                passat1, single_judge_info = 0.0, None
                        elif dataset=="mbpp":
                            if reeval:
                                passat1, single_judge_info = mbppeval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text[j],  sample["final_answer"], k)
                            else:
                                passat1, single_judge_info = 0.0, None
                        elif dataset=="livecodebench":
                            if reeval:
                                passat1, single_judge_info = 0.0, None
                            else:
                                passat1, single_judge_info = 0.0, None

                        passat1_list.append(passat1)
                        judge_info.append(single_judge_info)
                        # passat1 = sum(passat1_list)/len(passat1_list)

                    else:
                        raise ValueError("Unknown dataset: {}".format(dataset))

                    break
                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    time.sleep(0.5)

    # save result
        result = {
            "hyperparams": str(args),
            "prompt": sample["prompt"][0]["value"],
            "completion": decoded_text,
            "ground_truth": sample["final_answer"],
            "generated_tokens": generated_tokens_list[i*args.num_samples:(i+1)*args.num_samples],
            "avg_generated_tokens": sum(generated_tokens_list[i*args.num_samples:(i+1)*args.num_samples])/args.num_samples,
            "time": 0,
            "idx": idx,
            "n": args.num_samples,
            "finish_generation": finish_generation_list[i*args.num_samples:(i+1)*args.num_samples],
            "judge_info": judge_info,
            "passat1": sum(passat1_list)/len(passat1_list),
            "passat1_list": passat1_list
        }
        results.append(result)

    with open(results_file, "w") as f:
        results.sort(key=lambda x: x["idx"])
        json.dump(results, f, indent=4)
    
    # convert livecodebench format
    if dataset == "livecodebench":
        from convert_livecodebench import convert_json
        # convert convert_livecodebenchnch format
        results_file_converted = f"{args.output_dir}/results/{dataset}/{base_filename}_converted.json"
        convert_json(input_file=results_file, output_file=results_file_converted)
        if reeval:
            # need to cd to Livecodebench_pkg
            import subprocess
            import sys

            # Save current working directory
            orig_cwd = os.getcwd()
            lcb_pkg_dir = "LiveCodeBench_pkg"

            # Compose the command for custom_evaluator
            custom_eval_cmd = [
                sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
                "--custom_output_file", "../"+results_file_converted,
                "--release_version", "release_v5",
                "--start_date", "2024-08-01",
                "--num_process_evaluate", "1",
                "--timeout", "50"
            ]

            print("Running custom_evaluator for LiveCodeBench reeval (cd to Livecodebench_pkg first)...")
            try:
                os.chdir(lcb_pkg_dir)
                subprocess.run(custom_eval_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
            except Exception as e:
                print(f"Error running custom_evaluator: {e}", flush=True)
            finally:
                os.chdir(orig_cwd)
            livecodebench_results_file = f"{args.output_dir}/results/{dataset}/{base_filename}_converted_codegeneration_output_eval_all.json"
            with open(livecodebench_results_file, "r") as f:
                livecodebench_results = json.load(f)

            for r in results:
                for lcb_r in livecodebench_results:
                    if r["ground_truth"]["question_id"] == lcb_r["question_id"]:
                        r["passat1"] = lcb_r["pass@1"]
                        r["passat1_list"] = [int(passat1) for passat1 in lcb_r["graded_list"]]
                        r["judge_info"] = lcb_r["metadata"]
                        break
            with open(results_file, "w") as f:
                results.sort(key=lambda x: x["idx"])
                json.dump(results, f, indent=4)

    # calculate statistics
    total_num = len(results)
    pass_at_1 = sum([r["passat1"] for r in results]) / total_num if total_num > 0 else 0

    end_time = time.time()
    print("end", flush=True)
    print(f"Time taken: {(end_time - start_time)/3600} hours", flush=True)


    results_statistics = {
        "total_num": total_num,
        "pass@1": pass_at_1,
        "avg_token_length-all": sum([r["avg_generated_tokens"] for r in results]) / total_num if total_num > 0 else 0,
        "avg_token_length-correct": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0]) / len([r["passat1"] for r in results if r["passat1"] > 0]) if len([r["passat1"] for r in results if r["passat1"] > 0]) > 0 else 0,
        "time_taken/h": (end_time - start_time)/3600
    }

    all_idx = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
    results_statistics["all_idx"] = {i:j for i,j in all_idx}

    # save results_statistics
    with open(results_statistics_file, "w") as f:
        json.dump(results_statistics, f, indent=4)

    # push results_statistics to huggingface
    if args.push_results_to_hf:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=results_statistics_file,
            path_in_repo=results_statistics_file,
            repo_id=args.hf_repo_id,
            token=args.hf_token
        )
    print(results_statistics, flush=True)
    

if __name__ == "__main__":
    main()
