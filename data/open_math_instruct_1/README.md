# Open Math Instruct 1

This is a dataset created with the permissive
[Mistral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model.

It aims to improve mathematics capabilities of the model.

## Description

The `OpenMathInstruct-1` dataset comprises several fields as outlined below:

- `question`: The original question sourced from either the GSM8K or MATH training set.
- `generated_solution`: This field contains a synthetically generated solution that incorporates both textual reasoning and code blocks.
- `expected_answer`: The original dataset's provided ground-truth answer.
- `predicted_answer`: The answer predicted by the Mixtral model within the generated solution, typically extracted from `\boxed{}` notation.
- `error_message`: This field captures error states; it is set to `<not_executed>` if code execution was not required. Otherwise, it may be empty or contain a Python exception message from the evaluated code block. A "timeout" message indicates that the code block's execution exceeded 10 seconds, leading to an automatic halt of generation for any error or timeout occurrences.
- `is_correct`: Indicates whether the generated solution's final answer was deemed correct according to our grading script.
- `dataset`: Specifies the source dataset for the question, which can be either `gsm8k` or `math`.
- `generation_type`: Denotes the method used for generating the solution, categorized as either `without_reference_solution` or `masked_reference_solution`.

## Download Dataset

Run the get datasets script to download the data

```sh
bash get_datasets.sh
```

The above will create `train.txt` and `validation.txt` files ready for
tokenization.

## Tokenizate files

Finally run the `prepare.py` script on the separate files to permit creation of
the train.bin and validation.bin files needed for training.

For example for tiktoken:
```sh
python3 prepare.py -s -t train.txt -v validation.txt
```

Or for sentence piece (warning requires *a lot* of ram):
```sh
python3 prepare.py -s -t train.txt -v validation.txt --method sentencepiece
```

## Reference and more information

- https://huggingface.co/datasets/nvidia/OpenMathInstruct-1
- https://github.com/Kipok/NeMo-Skills/tree/main
- [Arxiv Paper: OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)

## Full text of dataset license

```
NVIDIA License

1. Definitions

“Licensor” means any person or entity that distributes its Work.
“Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works thereof that are made available under this license.
The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.

2. License Grant. Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.

3.3 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.

3.4 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.

3.5 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.

4. Disclaimer of Warranty. THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 

5. Limitation of Liability. EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```
