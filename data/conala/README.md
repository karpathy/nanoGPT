# Conala

These are scripts compatible with the Conala english-python dataset.

## Description

The Conala dataset comprises web available python snippets paired with
english instructions.

## Get dataset

The following script will allow for creation of the input.txt:

```bash
bash get_dataset.sh
```

Dive into the script to adjust prefixes, currently we have:

"#U:\n" for the user and "#B:\n" for our bot.

## Links

Huggingface Link:
https://huggingface.co/datasets/neulab/conala

Paper about curating and creating the dataset:
https://arxiv.org/abs/1805.08949

## Citation

```
@inproceedings{yin2018learning,
  title={Learning to mine aligned code and natural language pairs from stack overflow},
  author={Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, Graham},
  booktitle={2018 IEEE/ACM 15th international conference on mining software repositories (MSR)},
  pages={476--486},
  year={2018},
  organization={IEEE}
}
```

