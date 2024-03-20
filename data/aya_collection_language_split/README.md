# Aya Collection Language Split

This folder contains scripts compatible with the Aya Collection Language Split

The Aya Collection is a massive multilingual collection consisting of 513
million instances of prompts and completions covering a wide range of tasks.
This dataset is a language-split version of the Aya Collection, where the data
is organized by language instead of dataset.

## Download

To download the dataset, run the following command:

```bash
python3 get_dataset --url <url-to-split>
```

This will download the dataset and extract it to the current directory.

Note: Running without --url flag defaults to english split

## License

Dataset was released by Cohere For AI via the Apache 2.0 License.

## Citation

The following citation was provided on the Huggingface Website

```
@misc{singh2024aya,
      title={Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning}, 
      author={Shivalika Singh and Freddie Vargus and Daniel Dsouza and Börje F. Karlsson and Abinaya Mahendiran and Wei-Yin Ko and Herumb Shandilya and Jay Patel and Deividas Mataciunas and Laura OMahony and Mike Zhang and Ramith Hettiarachchi and Joseph Wilson and Marina Machado and Luisa Souza Moura and Dominik Krzemiński and Hakimeh Fadaei and Irem Ergün and Ifeoma Okoh and Aisha Alaagib and Oshan Mudannayake and Zaid Alyafeai and Vu Minh Chien and Sebastian Ruder and Surya Guthikonda and Emad A. Alghamdi and Sebastian Gehrmann and Niklas Muennighoff and Max Bartolo and Julia Kreutzer and Ahmet Üstün and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2402.06619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Hugginface Dataset Website

https://huggingface.co/datasets/CohereForAI/aya_collection_language_split
