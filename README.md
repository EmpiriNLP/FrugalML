# Frugal ML

This project is part of the COMP0087: Statistical Natural Language Processing module at University College London (UCL). The primary objective is to compare the computational resources required by different combinations of training techniques and models to achieve a specified performance level. Performance will be evaluated using Exact Match Accuracy (EmAcc), while computational cost will be measured in Floating Point Operations (FLOPs).

The first file to be checked in, inf3.py, has the following functionality:

- Sets up 4-bit quantization
- Loads meta-llama/Llama-3.1-8B-Instruct
- Pre-processes training data to build a prompt comprising the following:

    question = example["qa"].get("question", "No question available.")
    expected_answer = str(example['qa'].get("answer", "")).strip()  # Force text
    
    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "Return ONLY the final numerical answer with no text explanation\n\n"
        f"Pre Text Data:\n{pre_text}\n\n"
        f"Table Data:\n{table_str}\n\n"
        f"Post Text Data:\n{post_text}\n\n"
        f"Question: {question}\n"
        "Final Answer (number only): "
    )
    
- Note that this does not include any details of the "program" since this would represent a data leak.
- Processes train3.json (This is the first 22 entries of the train.json file).
- Processes both the expected answer and the predicted answer to maximize the likelihood of a match. Note that this is very much a hack just to get something working, and is one of the things that needs further thought (see next steps below).
- Displays the number of exact matches and percentage accuracy. Also displays total time and avg inference time.

Learnings/bug-fixes:

- The experiments were done on an Amazon EC2 g4dn.4xlarge server which has an NVidiai Tesla T$ GPU with 32Gb VRAM.
- Quantization was required to enable Llama-3.1-8B-Instruct to fit within VRAM.  The following quantization config was used:

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
    bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
    bnb_4bit_quant_type="nf4"  # NormalFloat4, best for Llama models
)

- It is normal for LLMs to return the prompt as part of the generated text. This boilerplate code was used to remove that.

    #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
- The regexp in clean answer was amended to remove the trailing ? which was matching all numbers.

    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    
Initial results
===============

Total Samples: 22
Correct Predictions: 12
Accuracy: 54.55%
Average Similarity Score: 77.00%
Elapsed time: 37.31368660926819 seconds
Avg Inf time: 1.696076664057645 seconds


I tried several smaller models, with the same code.

They were all a lot faster, but the accuracy is terrible. e.g.

#MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

Total Samples: 22
Correct Predictions: 1
Accuracy: 4.55%
Average Similarity Score: 41.91%
Elapsed time: 4.691933631896973 seconds
Avg Inf time: 0.21326971054077148 seconds
    
Next steps
==========
This first check-in is very rough and was only submitted in case it was useful for the rest of the team.

The results are in no way a meaningful baseline, do at least demonstrate a model getting more than 50% of the questions broadly correct.

There are plenty of improvements needed!

Short term, I will tidy the code more and make it much easier to switch model and run the pipeline on different datasets.  I will also fix any other bugs I find. 

Longer term, I think the work breaks down into the following tasks:

- Refine Task / Loss Function. This is probably the next critical thing to finalise before we can move on to fine tuning. If we continue to pursue the idea of having the LLM emit the correct numerical answer directly, then we need to establish a way of matching the result in a way which will create a useful signal with which to fine-tune the model. Alternatively, we may want to explore the approach used by the papers we've been looking at, of creating an intermediate "program" instead.  

- Fine-tuning methodology.  Once we have a satisfactory Task / Loss Function and stable pipeline, we can move on to exploring fine-tuning methodologies, ideally using a framework which allows us to switch between the approaches. Start with QLORA.

- Cost Function.  This can be done last. It basically changes what we record on the x-axis, and allows us to compare the M+FT approaches on the basis of a cost versus accuracy trade-off, rather than accuracy alone.
  
