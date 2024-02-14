# Into the Unknown: Self-Learning Large Language Models

***Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko***

*Paper to be submitted to arXiv and ACL 2024*

## Results files

Refer to the upper-level directory's readme for details of the files in `open_generation`, `induced_generation`, `oracle_selected`, and `external_prompt`.

There are some additional files here:

- `ds_Intel_neural-chat-7b-v3-3_open.pickle`: An example of the dataset constructed from Knowledge Searching, taken from the Open Generation experiment with mistral-dpo (Intel/neural-chat-7b-v3-3). We do not include the full texts of the retrieved documents; however, the source URL of each retrieved document is provided.
- `check.py`: The script used to analyze `ds_Intel_neural-chat-7b-v3-3_open.pickle`.
- `ds_human_eval.xlsx`: The result of human evaluation on the full texts of top-1 documents in `ds_Intel_neural-chat-7b-v3-3_open.pickle`.
- `after_training_result.pickle`: The hallucination scoring result on Q_H (i.e., the prompts in `ds_Intel_neural-chat-7b-v3-3_open.pickle`) after model training on D_train. Check `open_generation/res_Intel_neural-chat-7b-v3-3_OpenGen.pickle` to see the hallucination scoring result before model training.
