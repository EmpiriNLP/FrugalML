# FinQA
## Structure of interest
```json
[
    {
        "pre_text": [line1, line2, ...], # Missing exists ~20 in train
        "post_text": [line1, "."x n, line2, ...], # Missing exists ~639 in train
        "file_name": "company/year/page",
        "id": "company/year/page-index",
        "table": [[row1col1, row1col2, ...], [row2col1, row2col2, ...], ...],
        "table_ori": ...,
        "qa": {
            "question": "The question",
            "exe_ans": "The answer", # Could be numeric or textual e.g. "yes" ~77 in train

            "program": "op1, op2",
            "program_re": "op2(op1)",
            "gold_inds": {"text_1": "", ...},
            "answer": ..., # Doesn't exist always
            ... # Others e.g. explanation, steps, ann_table_rows, ann_text_rows, tfidftopn, model_input

        }
    },
    ... # Others e.g. table_retrieved, text_retrieved, table_retrieved_all, text_retrieved_all
]

```