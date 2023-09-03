## T2T Generation for biomedical Causal Inference

This repository contains the code for the project "T2T Generation for biomedical Causal Inference". 

This works builds off of InferBERT, and we used some of their code from https://github.com/XingqiaoWang/DeepCausalPV-master.

In addition this repository contains inference code from https://github.com/facebookresearch/llama, and finetuning code from https://github.com/google-research/albert.

I have included outputs of each stage in the repository so raw results can be viewed in `causal_inference/causal_results` and these can be used to produced tables or causal trees in `results_analysis`. 

To reproduce the main results of our project, please see "Instructions to reproduce Llama Results with Extended Features".

### Requirements 

* Linux 64 bit
* CUDA for GPU accelerated training and inference
* A GPU (we used 1xNvidia RTX 6000) for GPU accelerated training and inference 
* Two GPUs totalling over 46GB combined GPU memory (we used 2xNvidia RTX 6000), for Llama Inference
* An OpenAI API key for ChatGPT annotation 
* Conda for python environment management
* Two different python environments
  1. Pytorch (tested on 2.0.0+cu117) and Huggingface Transformers. This environment can be installed with `conda create --name torch --file requirements_torch.txt`
  2. Tensorflow 1.15. This environment can be installed with `conda create --name tf --file requirements_tf.txt`
* Pre-trained ALBERT-base model, originally from https://github.com/google-research/albert  to reproduce ALBERT finetuning for endpoint prediction
* Pre-trained Llama-2-chat 13B from https://github.com/facebookresearch/llama to reproduce LLM T2T generations
* The Analgesics-Induced Liver Failure data set from https://github.com/XingqiaoWang/DeepCausalPV-master 

For Convenience, we uploaded the raw dataset, ALBERT-base model, GPT-supervised BART model and template-supervised BART model to figshare. These can be found at https://doi.org/10.6084/m9.figshare.24077526

### Instructions to reproduce control results (original InferBERT liver failure study as presented by https://github.com/XingqiaoWang/DeepCausalPV-master)

1. Run `conda activate tf` to use the tensorflow environment
2. Run `preprocessing/./original_preprocessing.sh` to preprocess the dataset 
3. Put the ALBERT-base model files in `albert_finetuning/albert_base_model`
4. In the file `albert_finetuning/run_albert.sh` change `DATA_DIR` to `../preprocessing/original_proc` and change `OUPUT_DIR` to `albert_output/control` 
5. Run `albert_finetuning/./run_albert.sh` to finetune ALBERT as an endpoint prediction model, and make its predictions. This requires a GPU and takes a long time (around 2.5 hours in my case)
6. In the file `causal_infernece/original_causal_inference.sh` change `ALBERT_OUTPUT_NAME` to `control`
7. Run `causal_inference/./original_causal_inference.sh` to use the prediction results to perform causal inference. This outputs the raw results in `causal_inference/causal_results/control`
8. Analyse the results with the scripts in `results_analysis`

### Instructions to reproduce template-supervised T2T results

1. Run `conda activate torch` to use the pytorch environment
2. Run `preprocessing/./original_preprocessing.sh` to preprocess the dataset 
3. Run `t2t/training_examples/./generate_template_examples.sh` . By default this annotates 100 examples using the template. The number of examples can be changes in bash file.
4. Run `t2t/finetuning/./template_supervised_finetuning.sh` to use the annotated training examples to finetune the BART seq2seq model (GPU recommended)
5. Run `t2t/generation/seq2seq/./template_supervised_generation.sh` to use the finetuned BART model to generate sentences for all the preprocessed reports. This step takes a long time (2.5 hours in my case). GPU recommended.
6. Put the ALBERT-base model files in `albert_finetuning/albert_base_model` 
7. Run `conda activate tf` to use the tensorflow environment
8. In the file `albert_finetuning/run_albert.sh` change `DATA_DIR` to `../t2t/generation/seq2seq/template_supervised_generations` and change `OUPUT_DIR` to `albert_output/template_supervised` 
9. Run `albert_finetuning/./run_albert.sh` to finetune ALBERT as an endpoint prediction model, and make its predictions. This requires a GPU and takes a long time (around 2.5 hours in my case)
10. In the file `causal_inference/original_causal_inference.sh` change `ALBERT_OUTPUT_NAME` to `template_supervised`
11. Run `causal_inference/./original_causal_inference.sh` to use the prediction results to perform causal inference. This outputs the raw results in `causal_inference/causal_results/template_supervised`
12. Analyse the results with the scripts in `results_analysis`

### Instructions to reproduce LLM-supervised T2T results

1. Run `conda activate torch` to use the pytorch environment
2. Run `preprocessing/./original_preprocessing.sh` to preprocess the dataset 
3. In the file `t2t/training_examples/generate_gpt_examples.py` add the OpenAI API key on line 6
4. Run `t2t/training_examples/./generate_gpt_examples.sh` to extract and annotate 200 examples from the dataset for seq2seq finetuning
5. Run `t2t/finetuning/./gpt_supervised_finetuning.sh` to use the annotated training examples to finetune the BART seq2seq model (GPU recommended)
6. Run `t2t/generation/seq2seq/./gpt_supervised_generation.sh` to use the finetuned BART model to generate sentences for all the preprocessed reports. This step takes a long time (2.5 hours in my case). GPU recommended.
7. Put the ALBERT-base model files in `albert_finetuning/albert_base_model` 
8. Run `conda activate tf` to use the tensorflow environment
9. In the file `albert_finetuning/run_albert.sh` change `DATA_DIR` to `../t2t/generation/seq2seq/gpt_supervised_generations` and change `OUPUT_DIR` to `albert_output/gpt_supervised` 
10. Run `albert_finetuning/./run_albert.sh` to finetune ALBERT as an endpoint prediction model, and make its predictions. This requires a GPU and takes a long time (around 2.5 hours in my case)
11. In the file `causal_infernece/original_causal_inference.sh` change `ALBERT_OUTPUT_NAME` to `gpt_supervised`
12. Run `causal_inference/./original_causal_inference.sh` to use the prediction results to perform causal inference. This outputs the raw results in `causal_inference/causal_results/gpt_supervised`
13. Analyse the results with the scripts in `results_analysis`

### Instructions to reproduce Llama Results

1. Run `conda activate torch` to use the pytorch environment
2. Run `preprocessing/./original_preprocessing.sh` to preprocess the dataset
3. Put the Llama-2-chat 13B model files in `t2t/generation/llama/llama` 
4. Run `t2t/generation/llama/./llama_t2t.sh` to prompt llama to convert the preprocessed reports into sentences. This step requires the 2 GPUs mentioned in requirements. It takes a long time (around 10 hours in my case).
5. Put the ALBERT-base model files in `albert_finetuning/albert_base_model` 
6. Run `conda activate tf` to use the tensorflow environment
7. In the file `albert_finetuning/run_albert.sh` change `DATA_DIR` to `../t2t/generation/seq2seq/llama_generations` and change `OUPUT_DIR` to `albert_output/llama` 
8. Run `albert_finetuning/./run_albert.sh` to finetune ALBERT as an endpoint prediction model, and make its predictions this requires a GPU and takes a long time (around 2.5 hours in my case)
9. Run `causal_inference/./extended_causal_inference.sh` to use the prediction results to perform causal inference. This outputs the raw results in `causal_inference/causal_results/llama`
10. Analyse the results with the scripts in `results_analysis`

### Instructions to reproduce Llama Results with Extended Features

1. Run `conda activate torch` to use the pytorch environment
2. Run `preprocessing/./extended_preprocessing.sh` to preprocess the dataset with extended features
3. Put the Llama-2-chat 13B model files in `t2t/generation/llama/llama`
4. Run `t2t/generation/llama/./llama_extended_t2t.sh` to prompt llama to convert the pre-processed reports into sentences. This step requires the 2 GPUs mentioned in requirements. It takes a long time (around 10 hours in my case).
5. Put the ALBERT-base model files in `albert_finetuning/albert_base_model` 
6. Run `conda activate tf` to use the tensorflow environment
7. Run `albert_finetuning/./run_albert.sh` to finetune ALBERT as an endpoint prediction model, and make its predictions this requires a GPU and takes a long time (around 2.5 hours in my case)
8. Run `causal_inference/./extended_causal_inference.sh` to use the prediction results to perform causal inference. This outputs the raw results in `causal_inference/causal_results/llama_extended`
9. Analyse the results with the scripts in `results_analysis`

