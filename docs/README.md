<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Generating the documentation
# 生成文档

To generate the documentation, you have to build it. Several packages are necessary to build the doc.

要生成文档，你需要先构建它。构建文档需要安装几个包。

First, you need to install the project itself by running the following command at the root of the code repository:

首先，你需要在代码仓库的根目录运行以下命令来安装项目本身：

```bash
pip install -e .
```

You also need to install 2 extra packages:

你还需要安装 2 个额外的包：

```bash
# `hf-doc-builder` to build the docs
pip install git+https://github.com/huggingface/doc-builder@main
```

---
**NOTE** / **注意**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to commit the built documentation.

你只需要在本地检查文档时生成它（例如，如果你计划做更改并想在提交前查看效果）。你不需要提交已构建的文档。

---

## Building the documentation
## 构建文档

Once you have setup the `doc-builder` and additional packages with the pip install command above,
you can generate the documentation by typing the following command:

一旦你使用上面的 pip install 命令设置好 `doc-builder` 和其他包后，你可以通过输入以下命令来生成文档：

```bash
doc-builder build smolagents docs/source/en/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

你可以修改 `--build_dir` 来设置任何你喜欢的临时文件夹。这个命令会创建该文件夹并生成 MDX 文件，这些文件将在主网站上渲染为文档。你可以在你喜欢的 Markdown 编辑器中查看它们。

## Previewing the documentation
## 预览文档

To preview the docs, run the following command:

要预览文档，运行以下命令：

```bash
doc-builder preview smolagents docs/source/en/
```

The docs will be viewable at [http://localhost:5173](http://localhost:5173). You can also preview the docs once you
have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

文档将在 [http://localhost:5173](http://localhost:5173) 上可见。一旦你打开了 PR，你也可以预览文档。你会看到一个机器人添加评论，其中包含你的更改后的文档链接。

---
**NOTE** / **注意**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update
`_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

`preview` 命令只适用于现有的文档文件。当你添加一个全新的文件时，你需要更新 `_toctree.yml` 并重启 `preview` 命令（按 `ctrl-c` 停止它，然后再次调用 `doc-builder preview ...`）。

---

## Adding a new element to the navigation bar
## 向导航栏添加新元素

Accepted files are Markdown (.md).

接受的文件格式是 Markdown (.md)。

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/smolagents/blob/main/docs/source/_toctree.yml) file.

创建一个带扩展名的文件并将其放在 source 目录中。然后你可以通过在 [`_toctree.yml`](https://github.com/huggingface/smolagents/blob/main/docs/source/_toctree.yml) 文件中添加不带扩展名的文件名来将其链接到目录树。

## Renaming section headers and moving sections
## 重命名章节标题和移动章节

It helps to keep the old links working when renaming the section header and/or moving sections from one document to another. This is because the old links are likely to be used in Issues, Forums, and Social media and it'd make for a much more superior user experience if users reading those months later could still easily navigate to the originally intended information.

在重命名章节标题和/或将章节从一个文档移动到另一个文档时，保持旧链接有效是很有帮助的。这是因为旧链接可能会在 Issues、论坛和社交媒体中使用，如果几个月后阅读这些内容的用户仍然可以轻松导航到最初预期的信息，这将带来更好的用户体验。

Therefore, we simply keep a little map of moved sections at the end of the document where the original section was. The key is to preserve the original anchor.

因此，我们只需在原始章节所在文档的末尾保留一个移动章节的小地图。关键是保留原始锚点。

So if you renamed a section from: "Section A" to "Section B", then you can add at the end of the file:

```
Sections that were moved:

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```
and of course, if you moved it to another file, then:

```
Sections that were moved:

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

Use the relative style to link to the new file so that the versioned docs continue to work.

For an example of a rich moved section set please see the very end of [the transformers Trainer doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md).


## Writing Documentation - Specification
## 编写文档 - 规范

The `huggingface/smolagents` documentation follows the
[Google documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) style for docstrings,
although we can write them directly in Markdown.

`huggingface/smolagents` 文档遵循 [Google 文档](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 风格的 docstring，尽管我们可以直接用 Markdown 编写它们。

### Adding a new tutorial
### 添加新教程

Adding a new tutorial or section is done in two steps:

添加新教程或章节分两步完成：

- Add a new Markdown (.md) file under `./source`.
- 在 `./source` 下添加一个新的 Markdown (.md) 文件。
- Link that file in `./source/_toctree.yml` on the correct toc-tree.
- 在 `./source/_toctree.yml` 的正确目录树中链接该文件。

Make sure to put your new file under the proper section. If you have a doubt, feel free to ask in a Github Issue or PR.

确保将新文件放在正确的章节下。如果有疑问，请随时在 Github Issue 或 PR 中提问。

### Writing source documentation
### 编写源代码文档

Values that should be put in `code` should either be surrounded by backticks: \`like so\`. Note that argument names
and objects like True, None, or any strings should usually be put in `code`.

应该放在 `code` 中的值应该用反引号包围：\`像这样\`。注意参数名和对象（如 True、None 或任何字符串）通常应该放在 `code` 中。

When mentioning a class, function, or method, it is recommended to use our syntax for internal links so that our tool
adds a link to its documentation with this syntax: \[\`XXXClass\`\] or \[\`function\`\]. This requires the class or
function to be in the main package.

当提到类、函数或方法时，建议使用我们的内部链接语法，这样我们的工具会用这种语法添加到其文档的链接：\[\`XXXClass\`\] 或 \[\`function\`\]。这要求类或函数在主包中。

If you want to create a link to some internal class or function, you need to
provide its path. For instance: \[\`utils.ModelOutput\`\]. This will be converted into a link with
`utils.ModelOutput` in the description. To get rid of the path and only keep the name of the object you are
linking to in the description, add a ~: \[\`~utils.ModelOutput\`\] will generate a link with `ModelOutput` in the description.

如果你想创建到某个内部类或函数的链接，你需要提供其路径。例如：\[\`utils.ModelOutput\`\]。这将被转换为描述中带有 `utils.ModelOutput` 的链接。要去掉路径并只在描述中保留你链接到的对象名称，添加一个 ~：\[\`~utils.ModelOutput\`\] 将生成描述中带有 `ModelOutput` 的链接。

The same works for methods so you can either use \[\`XXXClass.method\`\] or \[~\`XXXClass.method\`\].

方法也是如此，所以你可以使用 \[\`XXXClass.method\`\] 或 \[~\`XXXClass.method\`\]。

#### Defining arguments in a method
#### 定义方法中的参数

Arguments should be defined with the `Args:` (or `Arguments:` or `Parameters:`) prefix, followed by a line return and
an indentation. The argument should be followed by its type, with its shape if it is a tensor, a colon, and its
description:

参数应该用 `Args:`（或 `Arguments:` 或 `Parameters:`）前缀定义，后跟换行和缩进。参数后面应该跟着它的类型，如果是张量则包括其形状，一个冒号，以及它的描述：

```
    Args:
        n_layers (`int`): The number of layers of the model.
```

If the description is too long to fit in one line, another indentation is necessary before writing the description
after the argument.

如果描述太长无法放在一行中，在参数后编写描述之前需要另一个缩进。

Here's an example showcasing everything so far:

```
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AlbertTokenizer`]. See [`~PreTrainedTokenizer.encode`] and
            [`~PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
```

For optional arguments or arguments with defaults we follow the following syntax: imagine we have a function with the
following signature:

```
def my_function(x: str = None, a: float = 1):
```

then its documentation should look like this:

```
    Args:
        x (`str`, *optional*):
            This argument controls ...
        a (`float`, *optional*, defaults to 1):
            This argument is used to ...
```

Note that we always omit the "defaults to \`None\`" when None is the default for any argument. Also note that even
if the first line describing your argument type and its default gets long, you can't break it on several lines. You can
however write as many lines as you want in the indented description (see the example above with `input_ids`).

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done between two lines of three backticks as usual in Markdown:


````
```
# first line of code
# second line
# etc
```
````

#### Writing a return block
#### 编写返回值块

The return block should be introduced with the `Returns:` prefix, followed by a line return and an indentation.
The first line should be the type of the return, followed by a line return. No need to indent further for the elements
building the return.

返回值块应该用 `Returns:` 前缀引入，后跟换行和缩进。第一行应该是返回值的类型，后跟换行。构成返回值的元素不需要进一步缩进。

Here's an example of a single value return:

```
    Returns:
        `List[int]`: A list of integers in the range [0, 1] --- 1 for a special token, 0 for a sequence token.
```

Here's an example of a tuple return, comprising several objects:

```
    Returns:
        `tuple(torch.FloatTensor)` comprising various elements depending on the configuration ([`BertConfig`]) and inputs:
        - ** loss** (*optional*, returned when `masked_lm_labels` is provided) `torch.FloatTensor` of shape `(1,)` --
          Total loss is the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        - **prediction_scores** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) --
          Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
```

#### Adding an image
#### 添加图片

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.

由于仓库快速增长，重要的是确保不添加会显著增加仓库大小的文件。这包括图片、视频和其他非文本文件。我们更倾向于利用 hf.co 托管的 `dataset`（如托管在 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) 上的那些）来放置这些文件并通过 URL 引用它们。我们建议将它们放在以下数据集中：[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。如果是外部贡献，请随意将图片添加到你的 PR 中，并请 Hugging Face 成员将你的图片迁移到此数据集。

#### Writing documentation examples
#### 编写文档示例

The syntax for Example docstrings can look as follows:

示例 docstring 的语法可以如下所示：

```
    Example:

    ```python
    >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'
    ```
```

The docstring should give a minimal, clear example of how the respective model
is to be used in inference and also include the expected (ideally sensible)
output.
Often, readers will try out the example before even going through the function
or class definitions. Therefore, it is of utmost importance that the example
works as expected.

docstring 应该给出一个最小的、清晰的示例，说明如何在推理中使用相应的模型，并包括预期的（理想情况下是合理的）输出。通常，读者会在查看函数或类定义之前尝试示例。因此，示例按预期工作是至关重要的。

